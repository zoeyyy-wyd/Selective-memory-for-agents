"""Injection format and answering (plan section 09). Injected entries carry an id, a timestamp and a
source type and are ordered by time. The answering model must answer only from injected content, cite
entry ids, and abstain when nothing supports an answer."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Protocol

from smem.llm import LLM
from smem.read import ReadResult
from smem.schemas import Entry, Fact

ABSTAIN_TEXT = "I don't know."
_CITE_RE = re.compile(r"\[((?:ep|f|fs)_[0-9a-f]{6,})\]")

ANSWER_SYSTEM = """You answer a question about a user using only the memory entries provided.
Rules:
- Use only the entries. Do not use outside knowledge and do not guess.
- Cite the ids of the entries you relied on in square brackets, e.g. [ep_1a2b3c4d5e].
- Entries are ordered by time. Facts may have a validity window "valid from X to Y"; pick the version that
  matches the time the question asks about. "now" means the latest version.
- If the entries do not contain the answer, reply exactly: I don't know.
- Be concise: one or two sentences."""


def format_entry(e: Entry) -> str:
    if isinstance(e, Fact):
        window = f"valid from {e.valid_from.date()}"
        if e.valid_to is not None:
            window += f" to {e.valid_to.date()}"
        return f"[{e.id}] ({window}; fact/{e.kind}; by {e.speaker}) {e.entity} {e.attribute}: {e.value}"
    return f"[{e.id}] ({e.ts.strftime('%Y-%m-%d')}; episode/{e.speaker}) {e.text}"


def format_context(entries: list[Entry]) -> str:
    ordered = sorted(entries, key=lambda e: e.ts)
    return "\n".join(format_entry(e) for e in ordered)


def check_citations(answer: str, injected_ids: set[str]) -> tuple[list[str], list[str]]:
    cited = _CITE_RE.findall(answer)
    valid = [c for c in cited if c in injected_ids]
    invalid = [c for c in cited if c not in injected_ids]
    return valid, invalid


class Answerer(Protocol):
    def answer(self, question: str, result: ReadResult, now: datetime) -> str: ...


class LLMAnswerer:
    def __init__(self, llm: LLM, max_tokens: int = 300):
        self.llm = llm
        self.max_tokens = max_tokens

    def answer(self, question: str, result: ReadResult, now: datetime) -> str:
        if result.abstain or not result.packed:
            return ABSTAIN_TEXT
        user = (f"Current date: {now.strftime('%Y-%m-%d')}\n\nMemory entries:\n{format_context(result.packed)}\n\n"
                f"Question: {question}")
        return self.llm.complete(ANSWER_SYSTEM, user, max_tokens=self.max_tokens).strip()


class ExtractiveAnswerer:
    """Offline stand-in: returns the most relevant injected entry with its citation."""

    def answer(self, question: str, result: ReadResult, now: datetime) -> str:
        if result.abstain or not result.packed:
            return ABSTAIN_TEXT
        best = max(result.packed, key=lambda e: (isinstance(e, Fact), result.packed_rel.get(e.id, 0.0)))
        if isinstance(best, Fact):
            return f"{best.value} [{best.id}]"
        return f"{best.text} [{best.id}]"
