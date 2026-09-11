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
