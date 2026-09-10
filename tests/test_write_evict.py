from datetime import timedelta

import numpy as np
import pytest

from smem.config import SystemConfig
from smem.schemas import Episode, ExtractionResult, Fact
from smem.system import SelectiveMemory
from tests.conftest import T0


def ep(i, text, day=0, tokens=10, entities=("user",)):
    return Episode(id=f"ep_{i:06x}aaaa", ts=T0 + timedelta(days=day), session_id=f"s{day}", turn_idx=0, speaker="user",
                   text=text, entities=list(entities), tokens=tokens)


def feed(mem: SelectiveMemory, episodes: list[Episode], facts: list[Fact] = (), day: int = 0):
    result = ExtractionResult(session_id=f"s{day}", episodes=episodes, facts=list(facts))
    vecs = mem.embedder.encode([e.text for e in episodes] + [f.text for f in facts])
    vec_of = {e.id: vecs[i] for i, e in enumerate(list(episodes) + list(facts))}
    return mem.writer.ingest(result, vec_of, T0 + timedelta(days=day))


@pytest.mark.parametrize("write,evict", [("all", "fifo"), ("all", "lru"), ("all", "random"), ("all", "utility_heuristic"),
                                         ("sieve", "swap"), ("sieve", "swap_no_hawkes"), ("novelty_threshold", "swap")])
def test_store_never_exceeds_budget(write, evict):
    cfg = SystemConfig().with_overrides({"budget.store_tokens": 60, "write.policy": write, "evict.policy": evict,
                                         "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    for i in range(25):
        feed(mem, [ep(i, f"event number {i} about topic {i % 5} with details {i * 7}", day=i)], day=i)
        for s in mem.writer.sieves:
            assert s.used <= 60
            assert s.used == mem.store.used_tokens(s.selected)
    assert mem.writer.active_ids() <= set(mem.store.ids())
    assert mem.writer.stats.evicted > 0 or mem.writer.stats.skipped_full > 0 or mem.writer.stats.skipped_threshold > 0


def test_fifo_evicts_oldest_first():
    cfg = SystemConfig().with_overrides({"budget.store_tokens": 30, "write.policy": "all", "evict.policy": "fifo",
                                         "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    for i in range(4):
        feed(mem, [ep(i, f"distinct event {i} alpha beta {i}", day=i)], day=i)
    ids = mem.writer.active_ids()
    assert ep(0, "").id not in ids and ep(3, "").id in ids


def test_lru_keeps_recently_read_entries():
    cfg = SystemConfig().with_overrides({"budget.store_tokens": 30, "write.policy": "all", "evict.policy": "lru",
                                         "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    for i in range(3):
        feed(mem, [ep(i, f"distinct event {i} alpha beta {i}", day=i)], day=i)
    mem.store.touch(ep(0, "").id, T0 + timedelta(days=10))
    feed(mem, [ep(3, "distinct event 3 alpha beta 3", day=3)], day=3)
    ids = mem.writer.active_ids()
    assert ep(0, "").id in ids and ep(1, "").id not in ids


def test_swap_prefers_evicting_redundant_entries():
    cfg = SystemConfig().with_overrides({"budget.store_tokens": 30, "write.policy": "all", "evict.policy": "swap",
                                         "write.hysteresis_gamma": 0.0, "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    feed(mem, [ep(0, "I adopted a cat named Milo yesterday", day=0), ep(1, "I adopted a cat called Milo", day=0),
               ep(2, "My employer is Northwind Labs in Boston", day=0)], day=0)
    assert len(mem.writer.active_ids()) == 3
    feed(mem, [ep(3, "I signed up for a marathon in June", day=1)], day=1)
    ids = mem.writer.active_ids()
    assert ep(2, "").id in ids and ep(3, "").id in ids
    assert sum(1 for i in (ep(0, "").id, ep(1, "").id) if i in ids) == 1  # one of the two duplicates went


def test_near_duplicate_episodes_are_merged_not_stored():
    cfg = SystemConfig().with_overrides({"write.policy": "all", "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    feed(mem, [ep(0, "I adopted a cat named Milo yesterday")])
    feed(mem, [ep(1, "I adopted a cat named Milo yesterday")], day=1)
    assert mem.writer.stats.merged == 1 and len(mem.writer.active_ids()) == 1


def test_fact_merge_unions_sources_and_update_builds_chain():
    cfg = SystemConfig().with_overrides({"write.policy": "all", "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    f1 = Fact(id="f_aaaaaa1", entity="user", attribute="city", value="Boston", valid_from=T0, sources=["ep_x"], tokens=4)
    f2 = Fact(id="f_aaaaaa2", entity="user", attribute="city", value="boston", valid_from=T0 + timedelta(days=3), sources=["ep_y"], tokens=4)
    f3 = Fact(id="f_aaaaaa3", entity="user", attribute="city", value="Seattle", valid_from=T0 + timedelta(days=30), sources=["ep_z"], tokens=4)
    feed(mem, [], [f1], day=0)
    feed(mem, [], [f2], day=3)
    assert mem.writer.stats.merged == 1 and mem.store.facts["f_aaaaaa1"].sources == ["ep_x", "ep_y"]
    feed(mem, [], [f3], day=30)
    chain = mem.store.chain("f_aaaaaa1")
    assert [f.value for f in chain] == ["Boston", "Seattle"] and chain[0].valid_to == f3.valid_from


def test_no_validity_chain_overwrites():
    cfg = SystemConfig().with_overrides({"write.policy": "all", "write.validity_chain": False,
                                         "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    f1 = Fact(id="f_bbbbbb1", entity="user", attribute="city", value="Boston", valid_from=T0, tokens=4)
    f3 = Fact(id="f_bbbbbb3", entity="user", attribute="city", value="Seattle", valid_from=T0 + timedelta(days=30), tokens=4)
    feed(mem, [], [f1], day=0)
    feed(mem, [], [f3], day=30)
    assert "f_bbbbbb1" not in mem.store and "f_bbbbbb3" in mem.writer.active_ids()


def test_chain_is_evicted_as_a_unit():
    cfg = SystemConfig().with_overrides({"budget.store_tokens": 40, "write.policy": "all", "evict.policy": "fifo",
                                         "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    f1 = Fact(id="f_cccccc1", entity="user", attribute="city", value="Boston", valid_from=T0, tokens=4)
    f2 = Fact(id="f_cccccc2", entity="user", attribute="city", value="Seattle", valid_from=T0 + timedelta(days=1), tokens=4)
    feed(mem, [], [f1], day=0)
    feed(mem, [], [f2], day=1)
    feed(mem, [ep(9, "a completely different long event about a marathon in June", day=2, tokens=35)], day=2)
    ids = mem.writer.active_ids()
    assert ("f_cccccc1" in ids) == ("f_cccccc2" in ids)


def test_sieve_active_store_is_the_best_sieve():
    cfg = SystemConfig().with_overrides({"budget.store_tokens": 80, "write.policy": "sieve", "evict.policy": "swap",
                                         "consolidation.policy": "no_consolidation"})
    mem = SelectiveMemory(cfg)
    for i in range(12):
        feed(mem, [ep(i, f"event {i} about {['cats', 'work', 'food', 'travel'][i % 4]} with detail {i * 3}", day=i)], day=i)
    active = mem.writer.active()
    assert all(active.value() >= s.value() - 1e-9 for s in mem.writer.sieves)
    assert len(mem.writer.sieves) > 1
    assert np.isfinite(active.value())
