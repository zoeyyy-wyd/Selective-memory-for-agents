"""Naive RAG: chunk sessions by turn, hybrid BM25 + dense (RRF), top-k until the read budget is full.
No write policy, no eviction, no chains: the horizontal line in Figure 1."""

from __future__ import annotations

from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

from baselines import BaselineOutput, answer_from_context
from evals.longmemeval.data import LMEQuestion
from smem.embed import Embedder, tokenize
from smem.llm import LLM
from smem.tokens import count_tokens


class NaiveRAGBaseline:
    name = "naive_rag"

    def __init__(self, llm: LLM | None, embedder: Embedder, read_tokens: int = 2000, k: int = 30,
                 max_chunk_tokens: int = 200):
        self.llm = llm
        self.embedder = embedder
        self.read_tokens = read_tokens
        self.k = k
        self.max_chunk_tokens = max_chunk_tokens

    def _chunks(self, q: LMEQuestion) -> list[tuple[str, int]]:
        out = []
        for s in q.sessions:
            for i, t in enumerate(s.turns):
                text = t.content.strip()
                if not text:
                    continue
                words = text.split()
                step = max(1, int(self.max_chunk_tokens * 0.75))
                for start in range(0, len(words), step):
                    piece = " ".join(words[start : start + step])
                    line = f"[{s.session_id}#{i}] ({s.ts.strftime('%Y-%m-%d')}; {t.role}) {piece}"
                    out.append((line, count_tokens(line)))
        return out

    def answer_question(self, q: LMEQuestion) -> BaselineOutput:
        chunks = self._chunks(q)
        if not chunks:
            return BaselineOutput("I don't know.", 0, 0)
        texts = [c[0] for c in chunks]
        bm25 = BM25Okapi([tokenize(t) or ["<empty>"] for t in texts])
        bm25_scores = bm25.get_scores(tokenize(q.question))
        vecs = self.embedder.encode(texts)
        qvec = self.embedder.encode([q.question])[0]
        dense = vecs @ qvec
        rrf: Counter[int] = Counter()
        for rank, idx in enumerate(np.argsort(-bm25_scores)[: self.k]):
            rrf[int(idx)] += 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(np.argsort(-dense)[: self.k]):
            rrf[int(idx)] += 1.0 / (60 + rank + 1)
        chosen, used = [], 0
        for idx, _ in rrf.most_common():
            if used + chunks[idx][1] <= self.read_tokens:
                chosen.append(idx)
                used += chunks[idx][1]
        chosen.sort()
        context = "\n".join(texts[i] for i in chosen)
        answer = answer_from_context(self.llm, q.question, context, q.question_date,
                                     max_tokens=getattr(self, "answer_max_tokens", 300))
        return BaselineOutput(answer, used, len(chosen), context)
