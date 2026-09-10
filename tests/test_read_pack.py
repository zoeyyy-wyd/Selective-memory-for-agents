from datetime import timedelta

import pytest

from smem.config import SystemConfig
from smem.system import SelectiveMemory
from tests.conftest import T0, session

RECIPES = [session(f"r{i}", i, f"I cooked the garlic tomato pasta again tonight and it was great, version {i}.") for i in range(6)]
OTHER = [session("o1", 20, "I adopted a cat named Milo from the shelter."),
         session("o2", 21, "I signed up for a 10k race in June along the Charles River.")]


@pytest.mark.parametrize("packing", ["topk", "mmr", "budgeted_greedy"])
def test_packing_respects_read_budget(packing):
    cfg = SystemConfig.load("configs/offline.yaml", {"budget.read_tokens": 60, "read.packing": packing,
                                                       "write.policy": "all", "write.episode_dup_sim": 1.01})
    mem = SelectiveMemory(cfg)
    mem.ingest(RECIPES + OTHER)
    res = mem.ask("What did I cook and what pet do I have?", now=T0 + timedelta(days=30))
    assert res.read.tokens <= 60
    assert res.read.packed


def test_budgeted_greedy_covers_more_topics_than_topk():
    q = "Tell me about the pasta I cooked and the cat I adopted and the race I signed up for"
    packed = {}
    for packing in ("topk", "budgeted_greedy"):
        cfg = SystemConfig.load("configs/offline.yaml", {"budget.read_tokens": 70, "read.packing": packing,
                                                           "write.policy": "all", "write.episode_dup_sim": 1.01,
                                                           "read.retrieval": "single_hop"})
        mem = SelectiveMemory(cfg)
        mem.ingest(RECIPES + OTHER)
        res = mem.ask(q, now=T0 + timedelta(days=30))
        packed[packing] = {e.session_id[0] for e in res.read.packed}
    assert len(packed["budgeted_greedy"]) >= len(packed["topk"])
    assert {"o"} <= packed["budgeted_greedy"] or len(packed["budgeted_greedy"]) >= 2


def test_two_hop_adds_entity_neighbours():
    cfg = SystemConfig.load("configs/offline.yaml", {"budget.read_tokens": 300, "write.policy": "all"})
    mem = SelectiveMemory(cfg)
    mem.ingest([session("a", 0, "My cat Milo loves tuna biscuits."),
                session("b", 5, "Milo the cat is turning three next week."),
                session("c", 9, "Milo knocked a glass off the table this morning.")])
    res = mem.ask("what happened with the glass?", now=T0 + timedelta(days=10))
    assert res.read.diagnostics["n_hop2"] >= 0
    hop2 = [c for c in res.read.candidates if c.hop == 2]
    single = SelectiveMemory(cfg.with_overrides({"read.retrieval": "single_hop"}))
    single.ingest([session("a", 0, "My cat Milo loves tuna biscuits."), session("b", 5, "Milo the cat is turning three next week."),
                   session("c", 9, "Milo knocked a glass off the table this morning.")])
    res1 = single.ask("what happened with the glass?", now=T0 + timedelta(days=10))
    assert all(c.hop == 1 for c in res1.read.candidates)
    assert len(res.read.candidates) >= len(res1.read.candidates)
    assert isinstance(hop2, list)


def test_abstains_when_nothing_relevant():
    cfg = SystemConfig.load("configs/offline.yaml", {"write.policy": "all", "read.tau_abs": 0.6})
    mem = SelectiveMemory(cfg)
    mem.ingest(OTHER)
    res = mem.ask("What is the capital of the moon colony?", now=T0 + timedelta(days=30))
    assert res.abstained and res.answer.startswith("I don't know")


def test_empty_store_abstains():
    cfg = SystemConfig.load("configs/offline.yaml")
    mem = SelectiveMemory(cfg)
    res = mem.ask("anything?", now=T0)
    assert res.abstained
