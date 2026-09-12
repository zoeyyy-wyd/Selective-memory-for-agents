"""Baselines (plan section 02). All use the same answering model and the same judge as the system;
only the context they build differs. `answer_question(q)` returns the answer plus what was injected."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from evals.longmemeval.data import LMEQuestion
from smem.answer import ABSTAIN_TEXT, ANSWER_SYSTEM
from smem.config import SystemConfig
from smem.embed import tokenize
from smem.llm import LLM


@dataclass
class BaselineOutput:
    answer: str
    injected_tokens: int
    injected_entries: int
    context: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Baseline(Protocol):
    name: str

    def answer_question(self, q: LMEQuestion) -> BaselineOutput: ...


def answer_from_context(llm: LLM | None, question: str, context: str, now: datetime,
                        max_tokens: int = 300) -> str:
    """LLM answer over a plain-text context; offline, the line sharing the most terms with the question."""
    if not context.strip():
        return ABSTAIN_TEXT
    if llm is None:
        q_terms = set(tokenize(question))
        best = max(context.splitlines(), key=lambda line: len(q_terms & set(tokenize(line))), default="")
        return best.strip() or ABSTAIN_TEXT
    user = f"Current date: {now.strftime('%Y-%m-%d')}\n\nMemory entries:\n{context}\n\nQuestion: {question}"
    return llm.complete(ANSWER_SYSTEM, user, max_tokens=max_tokens).strip()


def render_turns(q: LMEQuestion, session_filter: set[str] | None = None, only_evidence_turns: bool = False) -> str:
    lines = []
    for s in q.sessions:
        if session_filter is not None and s.session_id not in session_filter:
            continue
        for i, t in enumerate(s.turns):
            if only_evidence_turns and s.session_id in q.evidence_turns and i not in q.evidence_turns[s.session_id]:
                continue
            lines.append(f"[{s.session_id}#{i}] ({s.ts.strftime('%Y-%m-%d')}; {t.role}) {t.content.strip()}")
    return "\n".join(lines)


def build_baseline(name: str, cfg: SystemConfig, backend: str) -> Baseline:
    from smem.system import build_llm

    llm = (None if backend == "offline" else
           build_llm(cfg.models.answer_model, cfg.models.answer_base_url, cfg, provider=cfg.models.answer_provider,
                     api_key_env=cfg.models.answer_api_key_env))
    if name == "oracle":
        from baselines.oracle import OracleBaseline

        b = OracleBaseline(llm)
        b.answer_max_tokens = cfg.models.answer_max_tokens
        return b
    if name == "full_context":
        from baselines.full_context import FullContextBaseline

        b = FullContextBaseline(llm)
        b.answer_max_tokens = cfg.models.answer_max_tokens
        return b
    if name == "naive_rag":
        from baselines.naive_rag import NaiveRAGBaseline
        from smem.embed import get_embedder

        b = NaiveRAGBaseline(llm, get_embedder(cfg.models.embedder, cfg.models.embed_dim), cfg.budget.read_tokens)
        b.answer_max_tokens = cfg.models.answer_max_tokens
        return b
    if name == "mem0_oss":
        from baselines.mem0_oss import Mem0Baseline

        b = Mem0Baseline(llm, cfg)
        b.answer_max_tokens = cfg.models.answer_max_tokens
        return b
    raise ValueError(f"unknown baseline {name}")


