from datetime import timedelta

from smem.config import SystemConfig
from smem.schemas import Fact
from smem.system import SelectiveMemory
from tests.conftest import T0


def build(cfg_overrides=None, sessions=None):
    cfg = SystemConfig.load("configs/offline.yaml", {"budget.read_tokens": 200, **(cfg_overrides or {})})
    mem = SelectiveMemory(cfg)
    mem.ingest(sessions)
    return mem


def city_facts(res):
    return {e.value for e in res.read.packed if isinstance(e, Fact) and e.attribute == "location"}


def test_worked_example_now_march_and_whole_chain(worked_example):
    mem = build(sessions=worked_example)
    now = T0 + timedelta(days=120)
    assert city_facts(mem.ask("Which city does the user live in now?", now)) == {"Seattle"}
    assert city_facts(mem.ask("Where did I live in March?", now)) == {"Boston"}
    assert city_facts(mem.ask("How many times did I move?", now)) == {"Boston", "Seattle"}
    assert city_facts(mem.ask("Where did I live previously?", now)) == {"Boston", "Seattle"}
    chain = mem.store.chain(next(f.id for f in mem.store.facts.values() if f.value == "Boston"))
    assert [f.value for f in chain] == ["Boston", "Seattle"]
    assert chain[0].valid_to == chain[1].valid_from


def test_overwrite_ablation_loses_march(worked_example):
    mem = build({"write.validity_chain": False}, worked_example)
    res = mem.ask("Where did I live in March?", T0 + timedelta(days=120))
    assert city_facts(res) <= {"Seattle"}


def test_answer_cites_injected_ids(worked_example):
    mem = build(sessions=worked_example)
    res = mem.ask("Which city does the user live in now?", T0 + timedelta(days=120))
    assert res.cited_ids and not res.invalid_citations
    assert set(res.cited_ids) <= res.injected_ids


def test_evidence_bookkeeping(worked_example):
    mem = build(sessions=worked_example)
    evidence = mem.candidates_from_sessions({"s3", "s21"})
    assert evidence and evidence <= mem.writer.written_ids
    assert mem.candidates_from_sessions({"s3"}, {"s3": {0}})  # turn filter keeps turn-0 episodes and facts
    assert mem.surviving_ids() <= set(mem.store.ids())
    drift = mem.stats()["key_drift"]
    assert drift["n_keys"] >= 2 and drift["n_suspicious_attribute_pairs"] == 0


def test_budget_pressure_keeps_hot_chain(worked_example):
    mem = build({"budget.store_tokens": 70}, worked_example)
    assert mem.writer.active().used <= 70
    res = mem.ask("Which city does the user live in now?", T0 + timedelta(days=120))
    assert res.read.tokens <= 200
    assert mem.stats()["store_tokens"] <= 70


def test_persistent_store_roundtrip(tmp_path, worked_example):
    cfg = SystemConfig.load("configs/offline.yaml", {"write.policy": "all"})
    mem = SelectiveMemory(cfg, store_path=str(tmp_path / "m.sqlite"))
    mem.ingest(worked_example)
    n = len(mem.store)
    mem.close()
    from smem.store import MemoryStore

    reloaded = MemoryStore(mem.embedder.dim, str(tmp_path / "m.sqlite"))
    assert len(reloaded) == n


def test_ingest_cache_round_trip_is_exact(worked_example, tmp_path):
    """save_state after ingest, load_state into a fresh system: identical store, selection, evidence
    bookkeeping and read result. This is what lets read-side runs skip ingest entirely."""
    a = build(sessions=worked_example)
    a.save_state(tmp_path / "s")
    b = build(sessions=[])                 # fresh system, nothing ingested
    b.load_state(tmp_path / "s")
    assert b.writer.active_ids() == a.writer.active_ids()
    assert set(b.store.facts) == set(a.store.facts) and set(b.store.episodes) == set(a.store.episodes)
    assert b.writer.written_ids == a.writer.written_ids and b.candidate_origin == a.candidate_origin
    assert b.stats() == a.stats()
    now = T0 + timedelta(days=120)
    ra, rb = a.ask("Where did I live in March?", now), b.ask("Where did I live in March?", now)
    assert rb.injected_ids == ra.injected_ids and rb.answer == ra.answer
    chain = b.store.chain(next(f.id for f in b.store.facts.values() if f.value == "Boston"))
    assert [f.value for f in chain] == ["Boston", "Seattle"]      # chain pointers survived the round trip


def test_ingest_key_ignores_read_side_settings():
    from smem.config import SystemConfig

    base = SystemConfig.load("configs/offline.yaml")
    same = base.with_overrides({"read.packing": "topk", "budget.read_tokens": 500, "models.answer_model": "x",
                                "models.judge_model": "y", "read.tau_abs": 0.9})
    diff = base.with_overrides({"evict.policy": "fifo"})
    assert same.ingest_key("q1") == base.ingest_key("q1")
    assert diff.ingest_key("q1") != base.ingest_key("q1") and base.ingest_key("q2") != base.ingest_key("q1")


def test_raw_turns_reach_the_reader_and_survive_the_cache(worked_example, tmp_path):
    """write.keep_raw_turns stores every verbatim turn; budget.raw_tokens>0 hands the reader the source
    turns of the packed entries (neighbours included), under that budget, and they survive save/load."""
    now = T0 + timedelta(days=120)
    off = build(sessions=worked_example)
    assert off.ask("Where did I live in March?", now).read.excerpts == []      # default: off, identity unchanged
    on = build(sessions=worked_example, cfg_overrides={"write.keep_raw_turns": True, "budget.raw_tokens": 120})
    assert len(on.store.turns) == sum(len(s.turns) for s in worked_example)
    r = on.ask("Where did I live in March?", now).read
    assert r.excerpts and r.raw_tokens <= 120 and any("Boston" in t.text for t in r.excerpts)
    assert r.raw_tokens == sum(t.tokens for t in r.excerpts)
    on.save_state(tmp_path / "s")
    again = build(sessions=[], cfg_overrides={"write.keep_raw_turns": True, "budget.raw_tokens": 120})
    again.load_state(tmp_path / "s")
    assert again.store.turns == on.store.turns
    assert [(t.session_id, t.turn_idx) for t in again.ask("Where did I live in March?", now).read.excerpts] == \
           [(t.session_id, t.turn_idx) for t in r.excerpts]
    assert on.cfg.ingest_key("q") != off.cfg.ingest_key("q")                     # flipping it re-ingests
    assert on.cfg.with_overrides({"budget.raw_tokens": 0}).ingest_key("q") == on.cfg.ingest_key("q")  # read-side
