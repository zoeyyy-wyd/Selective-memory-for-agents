"""LongMemEval loading. Each question carries its own timestamped haystack of sessions plus the
evidence labels (answer_session_ids, turn-level has_answer) that make writing and forgetting
measurable. Files come from HuggingFace `xiaowu0162/longmemeval-cleaned` (MIT)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from smem.schemas import Session, Turn
from smem.temporal import parse_session_date
from smem.tokens import count_tokens

HF_REPO = "xiaowu0162/longmemeval-cleaned"
FILES = {"s": "longmemeval_s_cleaned.json", "m": "longmemeval_m_cleaned.json", "oracle": "longmemeval_oracle.json"}
QUESTION_TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
                  "multi-session", "temporal-reasoning", "knowledge-update"]


@dataclass
class LMEQuestion:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: datetime
    sessions: list[Session]                       # time-ordered
    evidence_session_ids: set[str]
    evidence_turns: dict[str, set[int]] = field(default_factory=dict)  # session id -> turns with has_answer

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def stratum(self) -> str:
        return f"{self.question_type}{'_abs' if self.is_abstention else ''}"

    @property
    def history_tokens(self) -> int:
        return sum(count_tokens(t.content) for s in self.sessions for t in s.turns)


def download_longmemeval(variant: str = "s", data_dir: str | Path = "data") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, FILES[variant], repo_type="dataset", local_dir=str(data_dir)))


def _parse_item(item: dict) -> LMEQuestion:
    sessions: list[Session] = []
    evidence_turns: dict[str, set[int]] = {}
    ids = item["haystack_session_ids"]
    dates = item["haystack_dates"]
    for sid, date, turns in zip(ids, dates, item["haystack_sessions"]):
        parsed_turns = []
        for i, t in enumerate(turns):
            role = t.get("role", "user")
            if role not in ("user", "assistant"):
                role = "assistant" if role == "system" else "user"
            has = bool(t.get("has_answer", False))
            parsed_turns.append(Turn(role=role, content=t.get("content", "") or "", has_answer=has))
            if has:
                evidence_turns.setdefault(sid, set()).add(i)
        sessions.append(Session(session_id=sid, ts=parse_session_date(date), turns=parsed_turns))
    sessions.sort(key=lambda s: s.ts)
    return LMEQuestion(
        question_id=item["question_id"], question_type=item["question_type"], question=item["question"],
        answer=str(item["answer"]), question_date=parse_session_date(item["question_date"]), sessions=sessions,
        evidence_session_ids=set(item.get("answer_session_ids", [])), evidence_turns=evidence_turns,
    )


def load_longmemeval(path: str | Path) -> list[LMEQuestion]:
    data = json.loads(Path(path).read_text())
    return [_parse_item(item) for item in data]


def count_unique_sessions(questions: list[LMEQuestion]) -> tuple[int, int]:
    """(unique sessions by content, total session slots). Filler sessions are shared across
    questions, so extraction is cached once per unique session."""
    seen = set()
    total = 0
    for q in questions:
        for s in q.sessions:
            total += 1
            seen.add(s.content_hash())
    return len(seen), total
