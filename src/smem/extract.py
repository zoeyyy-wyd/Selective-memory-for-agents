"""Session -> candidate episodes and facts (plan section 06, "Extraction").

Two backends behind one interface:
  * HeuristicExtractor: rule-based, no model. Used for the offline pipeline, tests and as the repair
    fallback when the LLM output violates the schema.
  * LLMExtractor: Qwen3-8B (or any OpenAI-compatible model) with schema-constrained decoding; results
    are cached by (session id, content hash, model, prompt version) so extraction runs once.
Relative dates in facts are normalised to absolute dates using the session timestamp."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from smem.llm import LLM, DiskCache, parse_json_object
from smem.schemas import Episode, ExtractionResult, Fact, Session, Turn, make_id
from smem.temporal import normalize_relative_date
from smem.tokens import count_tokens

PROMPT_VERSION = "v2"

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_idx": {"type": "integer"},
                    "speaker": {"type": "string", "enum": ["user", "assistant"]},
                    "text": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "date": {"type": ["string", "null"]},
                },
                "required": ["turn_idx", "speaker", "text", "entities"],
            },
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_idx": {"type": "integer"},
                    "entity": {"type": "string"},
                    "attribute": {"type": "string"},
                    "value": {"type": "string"},
                    "kind": {"type": "string", "enum": ["stated", "preference", "inferred"]},
                    "speaker": {"type": "string", "enum": ["user", "assistant"]},
                    "valid_from": {"type": ["string", "null"]},
                },
                "required": ["turn_idx", "entity", "attribute", "value", "kind", "speaker"],
            },
        },
    },
    "required": ["episodes", "facts"],
}

SYSTEM_PROMPT = """You extract long-term memory from one chat session between a user and an assistant.
Session date: {date}. Output JSON with two lists.

"episodes": one-sentence events worth remembering later (≤ 40 words each), mostly from the user's turns:
things the user did, plans, life events, problems, purchases, people and places. Skip generic chit-chat and
skip the assistant's explanations unless the user acted on them. Include the turn index and named entities.
Convert relative dates ("last Wednesday", "next month") to absolute dates using the session date.

"facts": stable attributes as (entity, attribute, value) triples, e.g. ("user", "city", "Seattle"),
("user", "allergy", "peanuts"), ("user", "favorite_cuisine", "Thai"). Use snake_case attribute names,
reuse attribute names consistently, entity "user" for the user. kind = "stated" for explicit statements,
"preference" for likes/dislikes, "inferred" if you had to infer it. speaker = who asserted it.
valid_from = ISO date if the fact explicitly starts at a date, else null."""

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|my|me|we|our|mine)\b", re.IGNORECASE)
_CAPITALISED = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
_STOP_ENTITIES = {"I", "The", "A", "An", "My", "Me", "We", "It", "This", "That", "You", "He", "She",
                  "They", "What", "How", "When", "Where", "Why", "Yes", "No", "Ok", "Okay", "Hi", "Hello",
                  "Thanks", "Thank", "Please", "Also", "And", "But", "So", "If", "In", "On", "At", "For"}

# (pattern, attribute, kind). Group 1 is the value. Kept deliberately small: the heuristic extractor
# exists to keep the pipeline runnable without a model, not to compete with it.
_FACT_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bI(?: just| recently)? moved to ([A-Z][\w .'-]+?)(?=[,.!?;]| this| last| for| with| and| because|$)"), "city", "stated"),
    (re.compile(r"\bI(?:'m| am)? (?:currently )?liv(?:e|ing) in ([A-Z][\w .'-]+?)(?=[,.!?;]| now| with| and| for|$)"), "city", "stated"),
    (re.compile(r"\bI (?:work|am working|started working) (?:at|for) ([A-Z][\w .&'-]+?)(?=[,.!?;]| as| in| now| and|$)"), "employer", "stated"),
    (re.compile(r"\bI(?:'m| am) (?:a|an) ([a-z][\w -]{2,30}?)(?= at| in| for| by| and|[,.!?;]|$)"), "occupation", "inferred"),
    (re.compile(r"\bI(?:'m| am) allergic to ([\w ,'-]+?)(?=[.!?;]| and| so| which|$)"), "allergy", "stated"),
    (re.compile(r"\b[Mm]y name is ([A-Z][\w'-]+)"), "name", "stated"),
    (re.compile(r"\b[Mm]y (?:favou?rite) ([a-z][\w ]{2,25}?) (?:is|are) ([\w ,'&-]+?)(?=[.!?;]| and| but| because|$)"), "favorite_{0}", "preference"),
    (re.compile(r"\bI (?:really )?(?:love|like|enjoy|prefer) ([\w ,'-]{3,40}?)(?=[.!?;]| and| but| because| so| when|$)"), "likes", "preference"),
    (re.compile(r"\bI (?:really )?(?:hate|dislike|can't stand|cannot stand) ([\w ,'-]{3,40}?)(?=[.!?;]| and| but| because| so|$)"), "dislikes", "preference"),
    (re.compile(r"\b[Mm]y ([a-z][\w' ]{2,20}?) is (?:called |named )?([A-Z][\w'-]+)"), "{0}", "stated"),
    (re.compile(r"\b[Mm]y birthday is (?:on )?([\w ]+?\d[\w ,]*?)(?=[.!?;]|$)"), "birthday", "stated"),
]


class Extractor(Protocol):
    def extract(self, session: Session) -> ExtractionResult: ...


def _entities_from(text: str) -> list[str]:
    ents = []
    for m in _CAPITALISED.finditer(text):
        cand = m.group(1).strip()
        if cand in _STOP_ENTITIES or len(cand) < 2:
            continue
        if cand not in ents:
            ents.append(cand)
    return ents[:8]


def _truncate(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    while words and count_tokens(" ".join(words)) > max_tokens:
        words = words[: max(1, int(len(words) * 0.8))]
        if len(words) == 1:
            break
    return " ".join(words)


def _clean_value(v: str) -> str:
    return v.strip().strip(".,;:!?\"'").strip()


class HeuristicExtractor:
    def __init__(self, max_episode_tokens: int = 60, assistant_episodes: bool = True):
        self.max_episode_tokens = max_episode_tokens
        self.assistant_episodes = assistant_episodes

    def extract(self, session: Session) -> ExtractionResult:
        episodes: list[Episode] = []
        facts: list[Fact] = []
        seen_fact_keys: set[tuple[str, str, str]] = set()
        for idx, turn in enumerate(session.turns):
            sentences = [s.strip() for s in _SENT_SPLIT.split(turn.content.strip()) if s.strip()]
            if turn.role == "assistant":
                if not self.assistant_episodes or not sentences:
                    continue
                # one episode per assistant turn: what the assistant suggested, first sentence only
                first = sentences[0]
                if len(first.split()) < 5:
                    continue
                text = _truncate(f"assistant suggested: {first}", self.max_episode_tokens)
                episodes.append(self._episode(session, idx, "assistant", text))
                continue
            for sent in sentences:
                if len(sent.split()) < 4:
                    continue
                if not _FIRST_PERSON.search(sent):
                    continue
                text = _truncate(sent, self.max_episode_tokens)
                episodes.append(self._episode(session, idx, "user", text))
                for pattern, attribute, kind in _FACT_PATTERNS:
                    m = pattern.search(sent)
                    if not m:
                        continue
                    groups = [_clean_value(g) for g in m.groups() if g]
                    if not groups:
                        continue
                    if "{0}" in attribute:
                        attr = attribute.format(re.sub(r"[^a-z0-9]+", "_", groups[0].lower()).strip("_"))
                        value = groups[1] if len(groups) > 1 else ""
                    else:
                        attr, value = attribute, groups[0]
                    if not value or len(value) > 60:
                        continue
                    key = ("user", attr, value.lower())
                    if key in seen_fact_keys:
                        continue
                    seen_fact_keys.add(key)
                    facts.append(self._fact(session, idx, "user", attr, value, kind, episodes[-1].id))
        return ExtractionResult(session_id=session.session_id, episodes=episodes, facts=facts, schema_ok=True)

    @staticmethod
    def _episode(session: Session, idx: int, speaker: str, text: str) -> Episode:
        ents = _entities_from(text)
        if speaker == "user" and "user" not in [e.lower() for e in ents]:
            ents.insert(0, "user")
        return Episode(
            id=make_id("ep", session.session_id, idx, text), ts=session.ts, session_id=session.session_id,
            turn_idx=idx, speaker=speaker, text=text, entities=ents, tokens=count_tokens(text),
        )

    @staticmethod
    def _fact(session: Session, idx: int, speaker: str, attr: str, value: str, kind: str, source: str) -> Fact:
        text = f"user {attr}: {value}"
        return Fact(
            id=make_id("f", session.session_id, "user", attr, value), entity="user", attribute=attr, value=value,
            valid_from=session.ts, sources=[source], kind=kind, speaker=speaker, session_id=session.session_id,
            tokens=count_tokens(text),
        )


class LLMExtractor:
    def __init__(self, llm: LLM, cache_dir: str | Path | None = None, constrained_decoding: bool = True,
                 max_episode_tokens: int = 60):
        self.llm = llm
        self.constrained_decoding = constrained_decoding
        self.max_episode_tokens = max_episode_tokens
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.fallback = HeuristicExtractor(max_episode_tokens)
        self.n_calls = 0
        self.n_schema_errors = 0

    def extract(self, session: Session) -> ExtractionResult:
        key = DiskCache.key("extract", PROMPT_VERSION, self.llm.model, self.constrained_decoding,
                            session.session_id, session.content_hash())
        raw = self.cache.get(key) if self.cache else None
        if raw is None:
            raw = self.llm.complete(
                SYSTEM_PROMPT.format(date=session.ts.strftime("%Y-%m-%d")),
                _render_session(session),
                json_schema=EXTRACTION_SCHEMA if self.constrained_decoding else None,
                max_tokens=2048,
            )
            self.n_calls += 1
            if self.cache:
                self.cache.put(key, raw)
        result = self.parse(session, raw)
        if not result.schema_ok:
            self.n_schema_errors += 1
        return result

    def parse(self, session: Session, raw: str) -> ExtractionResult:
        obj = parse_json_object(raw)
        if obj is None or not isinstance(obj.get("episodes"), list) or not isinstance(obj.get("facts"), list):
            fb = self.fallback.extract(session)
            return ExtractionResult(session_id=session.session_id, episodes=fb.episodes, facts=fb.facts,
                                    schema_ok=False, raw=raw)
        episodes: list[Episode] = []
        facts: list[Fact] = []
        ok = True
        n_turns = len(session.turns)
        for item in obj["episodes"]:
            try:
                idx = int(item["turn_idx"])
                idx = min(max(idx, 0), max(n_turns - 1, 0))
                speaker = item.get("speaker") or session.turns[idx].role
                if speaker not in ("user", "assistant"):
                    raise ValueError("bad speaker")
                text = _truncate(str(item["text"]).strip(), self.max_episode_tokens)
                if not text:
                    continue
                ts = session.ts
                if item.get("date"):
                    resolved = normalize_relative_date(str(item["date"]), session.ts)
                    ts = resolved or ts
                ents = [str(e) for e in item.get("entities", []) if str(e).strip()][:8]
                if speaker == "user" and "user" not in [e.lower() for e in ents]:
                    ents.insert(0, "user")
                episodes.append(Episode(
                    id=make_id("ep", session.session_id, idx, text), ts=ts, session_id=session.session_id,
                    turn_idx=idx, speaker=speaker, text=text, entities=ents, tokens=count_tokens(text)))
            except (KeyError, ValueError, TypeError):
                ok = False
        ep_by_turn: dict[int, str] = {}
        for ep in episodes:
            ep_by_turn.setdefault(ep.turn_idx, ep.id)
        for item in obj["facts"]:
            try:
                idx = int(item["turn_idx"])
                entity = str(item["entity"]).strip() or "user"
                attribute = re.sub(r"[^a-z0-9_]+", "_", str(item["attribute"]).strip().lower()).strip("_")
                value = str(item["value"]).strip()
                kind = item.get("kind", "stated")
                if kind not in ("stated", "preference", "inferred") or not attribute or not value:
                    raise ValueError("bad fact")
                speaker = item.get("speaker", "user")
                if speaker not in ("user", "assistant"):
                    speaker = "user"
                valid_from = session.ts
                if item.get("valid_from"):
                    resolved = normalize_relative_date(str(item["valid_from"]), session.ts)
                    valid_from = resolved or valid_from
                text = f"{entity} {attribute}: {value}"
                facts.append(Fact(
                    id=make_id("f", session.session_id, entity, attribute, value), entity=entity, attribute=attribute,
                    value=value, valid_from=valid_from, sources=[ep_by_turn[idx]] if idx in ep_by_turn else [],
                    kind=kind, speaker=speaker, session_id=session.session_id, tokens=count_tokens(text)))
            except (KeyError, ValueError, TypeError):
                ok = False
        return ExtractionResult(session_id=session.session_id, episodes=episodes, facts=facts, schema_ok=ok, raw=raw)


def _render_session(session: Session) -> str:
    lines = []
    for i, t in enumerate(session.turns):
        lines.append(f"[{i}] {t.role}: {t.content.strip()}")
    return "\n".join(lines)


def load_sessions_from_json(path: str | Path) -> list[Session]:
    """Demo / test input: [{"session_id": ..., "date": "2023/05/20 (Sat) 02:21", "turns": [{"role":..,"content":..}]}]"""
    from smem.temporal import parse_session_date

    data = json.loads(Path(path).read_text())
    out = []
    for s in data:
        out.append(Session(session_id=s["session_id"], ts=parse_session_date(s["date"]),
                           turns=[Turn(**t) for t in s["turns"]]))
    out.sort(key=lambda s: s.ts)
    return out


