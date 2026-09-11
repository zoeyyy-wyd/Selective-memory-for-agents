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
    assert ("location", "Boston", "stated") in triples
    assert ("favorite_cuisine", "Thai", "preference") in triples
    assert ("allergy", "peanuts", "stated") in triples
    assert all(e.speaker == "user" for e in r.episodes) and r.schema_ok
    assert all(f.sources for f in r.facts)


def test_llm_extractor_parses_constrained_output_and_normalises_dates():
    raw = json.dumps({
        "episodes": [{"turn_idx": 0, "speaker": "user", "text": "User moved to Boston", "entities": ["Boston"], "date": None}],
        "facts": [{"turn_idx": 0, "entity": "User", "attribute": "location", "value": "Boston", "kind": "stated",
                   "speaker": "user", "valid_from": "last Wednesday"}],
    })
    ex = LLMExtractor(ScriptedLLM([raw]))
    r = ex.extract(sess("I moved to Boston last Wednesday."))
    assert r.schema_ok and len(r.episodes) == 1 and len(r.facts) == 1
    f = r.facts[0]
    assert f.entity == "user" and f.attribute == "location" and f.sources == [r.episodes[0].id]
    assert f.valid_from < datetime(2023, 2, 4, 10) and f.valid_from.weekday() == 2


def test_llm_extractor_falls_back_on_malformed_output():
    ex = LLMExtractor(ScriptedLLM(["not json at all"]))
    r = ex.extract(sess("I just moved to Boston."))
    assert not r.schema_ok and ex.n_schema_errors == 1
    assert any(f.attribute == "location" for f in r.facts)  # heuristic fallback still produced the fact


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


def test_keys_long_tail_goes_into_entity_attribute_stays_canonical():
    from smem.extract import CANONICAL_ATTRIBUTES, canonical_entity

    raw = json.dumps({"episodes": [], "facts": [
        {"turn_idx": 0, "entity": "My French Press", "attribute": "other", "custom_attribute": "Coffee/Water Ratio",
         "value": "1:15", "kind": "stated", "speaker": "user"},
        {"turn_idx": 0, "entity": "the korean restaurants tried", "attribute": "count", "value": "5", "kind": "stated",
         "speaker": "user"},
    ]})
    r = LLMExtractor(ScriptedLLM([raw])).extract(sess("I switched my French press to 1:15. That makes five Korean places."))
    assert r.schema_ok
    keys = {(f.entity, f.attribute) for f in r.facts}
    assert keys == {("french_press", "coffee_water_ratio"), ("korean_restaurants_tried", "count")}
    assert canonical_entity("The Ethereal Dreams painting") == "ethereal_dreams_painting"
    assert "location" in CANONICAL_ATTRIBUTES and "other" in CANONICAL_ATTRIBUTES


def test_out_of_vocabulary_attribute_is_kept_but_counted_as_violation():
    raw = json.dumps({"episodes": [], "facts": [
        {"turn_idx": 0, "entity": "user", "attribute": "num_restaurants", "value": "5", "kind": "stated", "speaker": "user"},
        {"turn_idx": 0, "entity": "user", "attribute": "other", "custom_attribute": None, "value": "x", "kind": "stated",
         "speaker": "user"},
    ]})
    ex = LLMExtractor(ScriptedLLM([raw]), constrained_decoding=False)
    r = ex.extract(sess("hello"))
    assert not r.schema_ok and ex.n_schema_errors == 1
    assert [(f.entity, f.attribute) for f in r.facts] == [("user", "num_restaurants")]  # "other" without a name is dropped


def test_key_drift_diagnostic_reports_but_does_not_merge():
    from datetime import datetime

    from smem.schemas import Fact
    from smem.store import key_drift

    t = datetime(2023, 1, 1)
    facts = [Fact(id="f1", entity="user", attribute="location", value="Boston", valid_from=t),
             Fact(id="f2", entity="user", attribute="current_location", value="Seattle", valid_from=t),
             Fact(id="f3", entity="user", attribute="allergy", value="peanuts", valid_from=t),
             Fact(id="f4", entity="rachel", attribute="employer", value="Acme", valid_from=t),
             Fact(id="f5", entity="rachel_colleague", attribute="employer", value="Globex", valid_from=t)]
    d = key_drift(facts)
    assert d["suspicious_attribute_pairs"] == [("user", "current_location", "location")]
    assert d["suspicious_entity_pairs"] == [("rachel", "rachel_colleague")]
    assert d["n_entities"] == 3 and d["n_keys"] == 5
