"""Data model (plan section 04). Episodes are timestamped events pointing back to turns; facts are
(entity, attribute, value) triples with validity chains pointing back to episodes."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Speaker = Literal["user", "assistant"]
FactKind = Literal["stated", "preference", "inferred"]


def make_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


class Turn(BaseModel):
    role: Speaker
    content: str
    has_answer: bool = False  # LongMemEval evidence label at turn level


class Session(BaseModel):
    session_id: str
    ts: datetime
    turns: list[Turn]

    def content_hash(self) -> str:
        h = hashlib.sha1()
        for t in self.turns:
            h.update(t.role.encode())
            h.update(b"\x00")
            h.update(t.content.encode("utf-8"))
            h.update(b"\x01")
        return h.hexdigest()[:16]


class Episode(BaseModel):
    id: str
    ts: datetime
    session_id: str
    turn_idx: int
    speaker: Speaker
    text: str
    entities: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list, repr=False)
    tokens: int = 0
    consolidated: bool = False
    access_count: int = 0
    last_access: datetime | None = None

    @property
    def source_type(self) -> str:
        return f"episode/{self.speaker}"


class Fact(BaseModel):
    id: str
    entity: str
    attribute: str
    value: str
    valid_from: datetime
    valid_to: datetime | None = None
    superseded_by: str | None = None
    sources: list[str] = Field(default_factory=list)
    kind: FactKind = "stated"
    speaker: Speaker = "user"  # who asserted it; used by the volatility rule (section 07)
    session_id: str = ""      # session the fact was first extracted from; used by the evidence metrics
    embedding: list[float] = Field(default_factory=list, repr=False)
    tokens: int = 0
    access_count: int = 0
    last_access: datetime | None = None

    @property
    def entities(self) -> list[str]:
        return [self.entity]

    @property
    def text(self) -> str:
        return f"{self.entity} {self.attribute}: {self.value}"

    @property
    def source_type(self) -> str:
        return f"fact/{self.kind}"

    @property
    def ts(self) -> datetime:
        return self.valid_from

    def is_valid_at(self, t: datetime) -> bool:
        if t < self.valid_from:
            return False
        return self.valid_to is None or t < self.valid_to


Entry = Episode | Fact


class Budget(BaseModel):
    store_tokens: int = 10**9   # B; the default is effectively unbounded
    read_tokens: int = 2000     # R
    consolidate_every: int = 5  # N sessions, used only by the fixed_interval control


class ExtractionResult(BaseModel):
    """Output of the extractor for one session. schema_ok=False means the raw model output violated
    the schema and a repair or fallback was used; this feeds the write-error-rate metric."""

    session_id: str
    episodes: list[Episode]
    facts: list[Fact]
    schema_ok: bool = True
    raw: str | None = None
