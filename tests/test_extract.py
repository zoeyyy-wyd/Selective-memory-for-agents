import json
from datetime import datetime

from smem.extract import HeuristicExtractor, LLMExtractor
from smem.llm import ScriptedLLM, parse_json_object
from smem.schemas import Session, Turn


def sess(*contents):
    return Session(session_id="s1", ts=datetime(2023, 2, 4, 10), turns=[Turn(role="user", content=c) for c in contents])


def test_heuristic_extractor_finds_facts_and_episodes():
    r = HeuristicExtractor().extract(sess("I just moved to Boston, renting near Kendall. My favorite cuisine is Thai and I am allergic to peanuts."))
    triples = {(f.attribute, f.value, f.kind) for f in r.facts}
    assert ("city", "Boston", "stated") in triples
    assert ("favorite_cuisine", "Thai", "preference") in triples
    assert ("allergy", "peanuts", "stated") in triples
    assert all(e.speaker == "user" for e in r.episodes) and r.schema_ok
    assert all(f.sources for f in r.facts)


def test_llm_extractor_parses_constrained_output_and_normalises_dates():
    raw = json.dumps({
        "episodes": [{"turn_idx": 0, "speaker": "user", "text": "User moved to Boston", "entities": ["Boston"], "date": None}],
        "facts": [{"turn_idx": 0, "entity": "user", "attribute": "City", "value": "Boston", "kind": "stated",
                   "speaker": "user", "valid_from": "last Wednesday"}],
    })
    ex = LLMExtractor(ScriptedLLM([raw]))
    r = ex.extract(sess("I moved to Boston last Wednesday."))
    assert r.schema_ok and len(r.episodes) == 1 and len(r.facts) == 1
    f = r.facts[0]
    assert f.attribute == "city" and f.sources == [r.episodes[0].id]
    assert f.valid_from < datetime(2023, 2, 4, 10) and f.valid_from.weekday() == 2


def test_llm_extractor_falls_back_on_malformed_output():
    ex = LLMExtractor(ScriptedLLM(["not json at all"]))
    r = ex.extract(sess("I just moved to Boston."))
    assert not r.schema_ok and ex.n_schema_errors == 1
    assert any(f.attribute == "city" for f in r.facts)  # heuristic fallback still produced the fact


def test_llm_extractor_cache_hits(tmp_path):
    llm = ScriptedLLM(['{"episodes": [], "facts": []}'])
    ex = LLMExtractor(llm, cache_dir=tmp_path)
    ex.extract(sess("hello there my friend"))
    ex.extract(sess("hello there my friend"))
    assert len(llm.calls) == 1 and ex.n_calls == 1


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": [1, 2]} hope this helps') == {"a": [1, 2]}
    assert parse_json_object("[1, 2]") is None
