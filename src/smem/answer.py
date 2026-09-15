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
_CITE_RE = re.compile(r"\[((?:ep|f|fs|raw)_[0-9a-f]{6,})\]")

# Answering-prompt variants. "strict" is the original; larger readers (gpt-4.1, gpt-4o) abstained on a
# quarter of answerable questions under it and scored 10 points below gpt-4.1-mini. "grounded" keeps the
# no-outside-knowledge rule but abstains only when nothing relevant was injected.
ANSWER_PROMPTS = {
    "strict": None,     # filled below with ANSWER_SYSTEM
    # "notes": strict, plus the reading recipe the failure analysis asked for: read the verbatim excerpts as
    # the primary source, enumerate before counting, resolve conflicts by time, and answer preference
    # questions as advice rather than by restating the preference.
    # "notes2": the counting / recommendation rules of "notes" without "trust the excerpt over the entry"
    # and "later overrides earlier", which made the reader copy relative dates ("yesterday") out of old
    # excerpts and cost 4 temporal questions.
    "notes2": """You answer a question about a user using only the memory entries and conversation excerpts provided.
Rules:
- Use only the provided material. Do not use outside knowledge and do not guess.
- Each excerpt header gives the date of that conversation. Words like "yesterday" or "last week" inside an
  excerpt are relative to that header date, never to the current date.
- Names, numbers, amounts, times and titles must be copied exactly from the material.
- If the question asks "how many", "how much" or "which ones", first list each distinct item with its date,
  then count or add them; count the same item mentioned in several places only once.
- Entries are ordered by time. Facts may have a validity window "valid from X to Y"; pick the version that
  matches the time the question asks about. "now" means the latest version.
- If the question asks what you should recommend, suggest or how you should respond, answer by giving that
  recommendation, tailored to what the user has said about themselves.
- Cite the ids of the entries you relied on in square brackets, e.g. [ep_1a2b3c4d5e].
- If the provided material does not contain the answer, reply exactly: I don't know.
- Be concise: one or two sentences, without the intermediate list.""",
    "notes": """You answer a question about a user using only the memory entries and conversation excerpts provided.
Rules:
- Use only the provided material. Do not use outside knowledge and do not guess.
- The conversation excerpts are verbatim; when an entry and an excerpt differ, trust the excerpt. Details
  such as names, numbers, amounts, dates and titles must be copied from the excerpts exactly.
- Before answering, silently note every entry or excerpt that bears on the question. If the question asks
  "how many", "how much" or "which ones", first list each distinct item with its date, then count or add
  them; count the same item mentioned in several places only once.
- Entries are ordered by time. Facts may have a validity window "valid from X to Y"; pick the version that
  matches the time the question asks about. "now" means the latest version, and later statements
  override earlier ones.
- If the question asks what you should recommend, suggest or how you should respond, answer by giving that
  recommendation, tailored to what the user has said about themselves.
- Cite the ids of the entries you relied on in square brackets, e.g. [ep_1a2b3c4d5e].
- If the provided material does not contain the answer, reply exactly: I don't know.
- Be concise: one or two sentences, without the intermediate notes.""",
    "grounded": """You answer a question about a user using the memory entries provided.
Rules:
- Base the answer on the entries. Do not use outside knowledge about the user.
- If the entries contain information that answers the question, even partially, give the best answer they
  support and cite the ids of the entries you relied on in square brackets, e.g. [ep_1a2b3c4d5e].
- Entries are ordered by time. Facts may have a validity window "valid from X to Y"; pick the version that
  matches the time the question asks about. "now" means the latest version.
- Only if none of the entries are relevant to the question, reply exactly: I don't know.
- Be concise: one or two sentences.""",
}

ANSWER_SYSTEM = """You answer a question about a user using only the memory entries provided.
Rules:
- Use only the entries. Do not use outside knowledge and do not guess.
- Cite the ids of the entries you relied on in square brackets, e.g. [ep_1a2b3c4d5e].
- Entries are ordered by time. Facts may have a validity window "valid from X to Y"; pick the version that
  matches the time the question asks about. "now" means the latest version.
- If the entries do not contain the answer, reply exactly: I don't know.
- Be concise: one or two sentences."""


ANSWER_PROMPTS["strict"] = ANSWER_SYSTEM


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


def format_excerpts(result: ReadResult) -> str:
    return "\n".join(f"[{t.id}] ({t.ts.strftime('%Y-%m-%d')}; {t.speaker}, verbatim) {t.text}"
                     for t in result.excerpts)


def check_citations(answer: str, injected_ids: set[str]) -> tuple[list[str], list[str]]:
    cited = _CITE_RE.findall(answer)
    valid = [c for c in cited if c in injected_ids]
    invalid = [c for c in cited if c not in injected_ids]
    return valid, invalid


class Answerer(Protocol):
    def answer(self, question: str, result: ReadResult, now: datetime) -> str: ...


class LLMAnswerer:
    def __init__(self, llm: LLM, max_tokens: int = 300, prompt_version: str = "strict"):
        self.llm = llm
        self.max_tokens = max_tokens
        self.system = ANSWER_PROMPTS[prompt_version]

    def answer(self, question: str, result: ReadResult, now: datetime) -> str:
        if result.abstain or not result.packed:
            return ABSTAIN_TEXT
        user = f"Current date: {now.strftime('%Y-%m-%d')}\n\nMemory entries:\n{format_context(result.packed)}\n\n"
        if result.excerpts:
            user += f"Conversation excerpts (verbatim source of the entries above):\n{format_excerpts(result)}\n\n"
        user += f"Question: {question}"
        return self.llm.complete(self.system, user, max_tokens=self.max_tokens).strip()


class ExtractiveAnswerer:
    """Offline stand-in: returns the most relevant injected entry with its citation."""

    def answer(self, question: str, result: ReadResult, now: datetime) -> str:
        if result.abstain or not result.packed:
            return ABSTAIN_TEXT
        best = max(result.packed, key=lambda e: (isinstance(e, Fact), result.packed_rel.get(e.id, 0.0)))
        if isinstance(best, Fact):
            return f"{best.value} [{best.id}]"
        return f"{best.text} [{best.id}]"
