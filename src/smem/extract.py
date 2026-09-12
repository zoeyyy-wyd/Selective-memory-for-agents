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

PROMPT_VERSION = "v3"

# Keys: the long tail lives in the ENTITY (a noun phrase naming the thing described), the ATTRIBUTE
# comes from this closed list so that constrained decoding pins it. Two facts about the same thing
# therefore land on the same (entity, attribute) key, which is what the validity chain needs.
CANONICAL_ATTRIBUTES: dict[str, str] = {
    "name": "what someone or something is called",
    "location": "where someone lives or where a thing is",
    "employer": "company or organisation someone works for",
    "occupation": "job title or role",
    "status": "current state, e.g. finished, pending, broken, single",
    "count": "how many so far; write the CURRENT TOTAL, never the increment",
    "frequency": "how often, e.g. twice a week",
    "duration": "how long, elapsed or planned",
    "schedule": "when: day of week or time",
    "personal_best": "best result so far",
    "amount": "a quantity of money or material",
    "price": "a cost",
    "model": "brand, model or type of an item",
    "method": "how something is done",
    "setting": "a configured value or ratio",
    "preference": "a liking or a preferred way of doing things",
    "favorite": "the favourite instance of a category",
    "dislike": "something disliked",
    "allergy": "an allergen",
    "health": "a health condition or symptom",
    "goal": "a target or plan",
    "relationship": "how two people are related",
    "date": "a date attached to the entity: birthday, anniversary, deadline",
    "contact": "phone, email or address",
    "other": "none of the above; put a snake_case name in custom_attribute",
}

_ENTITY_DETERMINER = re.compile(r"^(?:my|the|a|an|our|his|her|their)_")


def canonical_entity(name: str) -> str:
    """Deterministic entity normalisation: snake_case, no leading determiner. Never merges two
    different names; that decision is left to a human after reading the key-drift diagnostic."""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    s = _ENTITY_DETERMINER.sub("", s)
    return re.sub(r"_+", "_", s).strip("_") or "user"


def canonical_attribute(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())).strip("_")

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "episodes": {
            "type": "array", "maxItems": 40,
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
            "type": "array", "maxItems": 60,
            "items": {
                "type": "object",
                "properties": {
                    "turn_idx": {"type": "integer"},
                    "entity": {"type": "string"},
                    "attribute": {"type": "string", "enum": list(CANONICAL_ATTRIBUTES)},
                    "custom_attribute": {"type": ["string", "null"]},
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

"facts": (entity, attribute, value) triples for anything a later question could ask about.

Entity rules. The entity is THE THING THE ATTRIBUTE DESCRIBES. Use "user" only for the user's own personal
attributes (name, location, employer, occupation, allergy, health, preferences, relationships). Anything the
user owns, does, tracks or collects, and any other person, is its own entity: a short snake_case noun phrase,
e.g. "charity_5k_run", "korean_restaurants_tried", "ethereal_dreams_painting", "rachel", "french_press".
Reuse exactly the same entity name whenever the same thing comes up again.

Attribute rules. attribute MUST be one of: {attributes}. If nothing fits, use "other" and put a snake_case
name in custom_attribute. For counts and totals write the current total, never the increment.

Examples:
  ("user", "location", "Seattle")            ("user", "allergy", "peanuts")
  ("charity_5k_run", "personal_best", "24:10")   ("korean_restaurants_tried", "count", "5")
  ("ethereal_dreams_painting", "location", "living room")   ("rachel", "employer", "Acme Corp")
  ("french_press", "setting", "1 tbsp coffee per 150 ml water")

kind = "stated" for explicit statements, "preference" for likes/dislikes, "inferred" if you had to infer it.
speaker = who asserted it. valid_from = ISO date if the fact explicitly starts at a date, else null."""

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|my|me|we|our|mine)\b", re.IGNORECASE)
_CAPITALISED = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
_STOP_ENTITIES = {"I", "The", "A", "An", "My", "Me", "We", "It", "This", "That", "You", "He", "She",
                  "They", "What", "How", "When", "Where", "Why", "Yes", "No", "Ok", "Okay", "Hi", "Hello",
                  "Thanks", "Thank", "Please", "Also", "And", "But", "So", "If", "In", "On", "At", "For"}

# (pattern, attribute, kind). Group 1 is the value. Kept deliberately small: the heuristic extractor
# exists to keep the pipeline runnable without a model, not to compete with it.
_FACT_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bI(?: just| recently)? moved to ([A-Z][\w .'-]+?)(?=[,.!?;]| this| last| for| with| and| because|$)"), "location", "stated"),
    (re.compile(r"\bI(?:'m| am)? (?:currently )?liv(?:e|ing) in ([A-Z][\w .'-]+?)(?=[,.!?;]| now| with| and| for|$)"), "location", "stated"),
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
                 max_episode_tokens: int = 60, max_turn_tokens: int = 2000,
                 max_output_tokens: int = 4096, loop_retry_temperature: float = 0.6):
        self.llm = llm
        self.constrained_decoding = constrained_decoding
        self.max_episode_tokens = max_episode_tokens
        self.max_turn_tokens = max_turn_tokens
        self.max_output_tokens = max_output_tokens
        self.loop_retry_temperature = loop_retry_temperature
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.fallback = HeuristicExtractor(max_episode_tokens)
        self.n_calls = 0
        self.n_schema_errors = 0
        # attribute='other' with no custom_attribute: a softer violation than broken JSON,
        # counted apart so the headline error rate means 'output we could not use'.
        self.n_missing_custom_attr = 0
        self.n_truncated = 0
        self.n_looped = 0          # greedy output was degenerate
        self.n_loop_fixed = 0      # ...and the sampled retry was not

    def extract(self, session: Session) -> ExtractionResult:
        key = DiskCache.key("extract", PROMPT_VERSION, self.llm.model, self.constrained_decoding,
                            self.max_turn_tokens, session.session_id, session.content_hash())
        system = SYSTEM_PROMPT.format(date=session.ts.strftime("%Y-%m-%d"),
                                      attributes=", ".join(f'"{k}" ({v})' for k, v in CANONICAL_ATTRIBUTES.items()))
        user = _render_session(session, self.max_turn_tokens)
        schema = EXTRACTION_SCHEMA if self.constrained_decoding else None

        raw = self.cache.get(key) if self.cache else None
        if raw is None:
            raw = self.llm.complete(system, user, json_schema=schema, max_tokens=self.max_output_tokens)
            self.n_calls += 1
            if self.cache:
                self.cache.put(key, raw)
        # Greedy decoding is deterministic, which is what the cache relies on, but it also has
        # repetition attractors: 81 of 105 bad dev outputs were one value repeated to the cap.
        # A little temperature escapes them (0.3 did not, 0.6 did). Retry once, sampled, and cache
        # the retry in place of the degenerate greedy output -- the cache key stays the same, the
        # meta says which path produced it. Extraction is preprocessing shared by every ablation,
        # so a different sampler on 2% of sessions moves no comparison; it is still recorded.
        if self.loop_retry_temperature > 0 and looks_degenerate(raw, parse_json_object(raw)):
            self.n_looped += 1
            retry = self.llm.complete(system, user, json_schema=schema, max_tokens=self.max_output_tokens,
                                      temperature=self.loop_retry_temperature)
            self.n_calls += 1
            if not looks_degenerate(retry, parse_json_object(retry)):
                self.n_loop_fixed += 1
                raw = retry
                if self.cache:
                    self.cache.put(key, raw, {"retry_temperature": self.loop_retry_temperature})
        result = self.parse(session, raw)
        if not result.schema_ok:
            self.n_schema_errors += 1
        return result

    def parse(self, session: Session, raw: str) -> ExtractionResult:
        obj = parse_json_object(raw)
        truncated = obj is None and not raw.rstrip().endswith(("}", "]"))
        if truncated:
            # A body that does not close its own JSON hit the output cap -- on dev, 81 of 105 such
            # cases were the model repeating one value until the cap. Everything before the cut is
            # still well-formed under the grammar, so salvage it: a session that looped after
            # extracting 20 good facts should keep those 20, not fall back to the heuristic
            # extractor. Still counted as a schema error; salvage improves the data, not the stat.
            self.n_truncated += 1
            obj = salvage_truncated_json(raw)
        if obj is None or not isinstance(obj.get("episodes"), list) or not isinstance(obj.get("facts"), list):
            fb = self.fallback.extract(session)
            return ExtractionResult(session_id=session.session_id, episodes=fb.episodes, facts=fb.facts,
                                    schema_ok=False, raw=raw)
        ok = not truncated
        episodes: list[Episode] = []
        facts: list[Fact] = []
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
                value_raw = str(item["value"]).strip()
                entity = canonical_entity(str(item["entity"]))
                raw_attr = canonical_attribute(str(item["attribute"]))
                if raw_attr == "other":
                    attribute = canonical_attribute(str(item.get("custom_attribute") or ""))
                    if not attribute:
                        # The schema's enum offers "other" but cannot require custom_attribute
                        # alongside it, and the model does skip it. Dropping the fact loses it
                        # outright; keying it on "other" would be worse still, since a second
                        # "other" fact about the same entity would read as a knowledge update and
                        # open a bogus validity chain. Derive a distinct key from the value instead.
                        attribute = _attr_from_value(value_raw)
                        self.n_missing_custom_attr += 1
                elif raw_attr in CANONICAL_ATTRIBUTES:
                    attribute = raw_attr
                else:
                    # only reachable without constrained decoding: keep the fact, count the violation
                    attribute = raw_attr
                    ok = False
                value = value_raw
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


def looks_degenerate(raw: str, obj: dict[str, Any] | None, min_repeat: int = 5) -> bool:
    """A greedy decode that fell into a repetition attractor: either it ran to the cap without
    closing its JSON, or it closed (maxItems forces that) with one value repeated over and over.
    Clean dev outputs never repeat a value 5 times -- a count updated five times is five values."""
    if obj is None:
        return not raw.rstrip().endswith(("}", "]"))
    vals = [str(f.get("value", "")) for f in obj.get("facts", []) if isinstance(f, dict)]
    vals += [str(e.get("text", "")) for e in obj.get("episodes", []) if isinstance(e, dict)]
    counts: dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return bool(counts) and max(counts.values()) >= min_repeat


def salvage_truncated_json(raw: str) -> dict[str, Any] | None:
    """Recover the complete prefix of an extraction body cut mid-item. Cut back to the last item
    boundary and close whatever is open; both arrays are top-level and the grammar guarantees every
    item before the cut is complete, so a handful of closers covers the cases."""
    cut = raw.rfind("},")
    if cut == -1:
        cut = raw.rfind("}")
    if cut == -1:
        return None
    head = raw[: cut + 1]
    in_facts = '"facts"' in head
    for closer in ("]}", "]}}", '], "facts": []}', '], "facts": []}}'):
        if in_facts and "facts" in closer:
            continue
        try:
            obj = json.loads(head + closer)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("episodes"), list) and isinstance(obj.get("facts"), list):
            return obj
    return None


def _attr_from_value(value: str, max_words: int = 3) -> str:
    """Fallback key for a fact the model marked "other" without naming a custom_attribute. The
    leading words of the value keep two such facts about one entity on different keys."""
    words = re.findall(r"[a-z0-9]+", value.lower())[:max_words]
    return "_".join(words)[:40] or "other"


def _render_session(session: Session, max_turn_tokens: int | None = None) -> str:
    """Turns longer than max_turn_tokens (long assistant essays and code dumps) are cut in the middle;
    both ends are kept because user facts tend to sit at the start and follow-ups at the end."""
    lines = []
    for i, t in enumerate(session.turns):
        content = t.content.strip()
        if max_turn_tokens and count_tokens(content) > max_turn_tokens:
            words = content.split()
            keep = max(20, int(len(words) * max_turn_tokens / count_tokens(content)))
            half = keep // 2
            content = " ".join(words[:half]) + " [...] " + " ".join(words[-half:])
        lines.append(f"[{i}] {t.role}: {content}")
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


