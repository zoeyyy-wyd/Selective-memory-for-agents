"""Mem0 open-source baseline, deployed locally: extraction model = the local vLLM server (Qwen3-8B),
embedder = a local sentence-transformers model, vector store = in-process Qdrant. Requires
`pip install mem0ai` (the `baselines` extra). Sessions are added in time order; the question is answered
from `memory.search` results packed under the read budget."""

from __future__ import annotations

import tempfile

from baselines import BaselineOutput, answer_from_context
from evals.longmemeval.data import LMEQuestion
from smem.config import SystemConfig
from smem.llm import LLM
from smem.tokens import count_tokens


class Mem0Baseline:
    name = "mem0_oss"

    def __init__(self, llm: LLM | None, cfg: SystemConfig, top_k: int = 30):
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the baselines extra: pip install -e '.[baselines]'") from exc
        self.llm = llm
        self.cfg = cfg
        self.top_k = top_k
        self._Memory = Memory

    def _fresh(self):
        embedder = self.cfg.models.embedder if self.cfg.models.embedder != "hash" else "BAAI/bge-m3"
        config = {
            "llm": {"provider": "openai", "config": {"model": self.cfg.models.extract_model,
                                                     "openai_base_url": self.cfg.models.extract_base_url,
                                                     "api_key": "EMPTY", "temperature": 0.0}},
            "embedder": {"provider": "huggingface", "config": {"model": embedder}},
            "vector_store": {"provider": "qdrant", "config": {"path": tempfile.mkdtemp(prefix="mem0_"),
                                                              "on_disk": False}},
        }
        return self._Memory.from_config(config)

    def answer_question(self, q: LMEQuestion) -> BaselineOutput:
        memory = self._fresh()
        user_id = q.question_id
        n_add_calls = 0
        for s in q.sessions:
            messages = [{"role": t.role, "content": t.content} for t in s.turns if t.content.strip()]
            if messages:
                memory.add(messages, user_id=user_id, metadata={"session_id": s.session_id, "date": s.ts.isoformat()})
                n_add_calls += 1
        hits = memory.search(q.question, user_id=user_id, limit=self.top_k)
        results = hits.get("results", hits) if isinstance(hits, dict) else hits
        lines, used = [], 0
        for h in results:
            text = h.get("memory") if isinstance(h, dict) else str(h)
            meta = h.get("metadata") or {} if isinstance(h, dict) else {}
            line = f"[{h.get('id', '?')}] ({meta.get('date', '')[:10]}) {text}"
            cost = count_tokens(line)
            if used + cost > self.cfg.budget.read_tokens:
                break
            lines.append(line)
            used += cost
        context = "\n".join(lines)
        answer = answer_from_context(self.llm, q.question, context, q.question_date)
        return BaselineOutput(answer, used, len(lines), context, {"mem0_add_calls": n_add_calls})
