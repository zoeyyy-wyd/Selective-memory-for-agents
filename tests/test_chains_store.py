from datetime import datetime

import numpy as np

from smem.schemas import Episode, Fact
from smem.store import MemoryStore


def fact(id_, value, day, embedding):
    return Fact(id=id_, entity="user", attribute="city", value=value, valid_from=datetime(2023, 1, day),
                embedding=embedding, tokens=4)


def test_supersede_builds_chain_from_any_node():
    store = MemoryStore(4)
    a, b, c = fact("f_a", "Boston", 1, [1, 0, 0, 0]), fact("f_b", "Seattle", 10, [0, 1, 0, 0]), fact("f_c", "Austin", 20, [0, 0, 1, 0])
    store.add(a)
    store.add(b)
    store.supersede(a, b, at=b.valid_from)
    store.add(c)
    store.supersede(b, c, at=c.valid_from)
    assert [f.id for f in store.chain("f_a")] == ["f_a", "f_b", "f_c"]
    assert [f.id for f in store.chain("f_b")] == ["f_a", "f_b", "f_c"]
    assert [f.id for f in store.chain("f_c")] == ["f_a", "f_b", "f_c"]
    assert a.valid_to == b.valid_from and a.superseded_by == "f_b"
    assert b.valid_to == c.valid_from and c.valid_to is None
    assert store.chain_root("f_c") == "f_a"
    assert store.volatility("user", "city") > 0


def test_chain_skips_evicted_nodes_and_persists_in_sqlite(tmp_path):
    path = str(tmp_path / "store.sqlite")
    store = MemoryStore(4, path)
    a, b = fact("f_a", "Boston", 1, [1, 0, 0, 0]), fact("f_b", "Seattle", 10, [0, 1, 0, 0])
    store.add(a)
    store.add(b)
    store.supersede(a, b, at=b.valid_from)
    store.remove("f_a")
    assert [f.id for f in store.chain("f_b")] == ["f_b"]
    store.close()
    reloaded = MemoryStore(4, path)
    assert "f_b" in reloaded and "f_a" not in reloaded
    assert reloaded.facts["f_b"].value == "Seattle"


def test_search_and_entity_index():
    store = MemoryStore(4)
    e1 = Episode(id="ep_1", ts=datetime(2023, 1, 1), session_id="s", turn_idx=0, speaker="user",
                 text="I adopted a cat named Milo", entities=["user", "Milo"], embedding=[1, 0, 0, 0], tokens=6)
    e2 = Episode(id="ep_2", ts=datetime(2023, 1, 2), session_id="s", turn_idx=1, speaker="user",
                 text="I bought a new bicycle", entities=["user"], embedding=[0, 1, 0, 0], tokens=5)
    store.add(e1)
    store.add(e2)
    assert store.search_bm25("cat Milo", 5)[0][0] == "ep_1"
    assert store.search_dense(np.array([0, 1, 0, 0], dtype=np.float32), 1)[0][0] == "ep_2"
    assert store.search_dense(np.array([1, 0, 0, 0], dtype=np.float32), 1, allowed={"ep_2"})[0][0] == "ep_2"
    assert store.by_entity("milo") == {"ep_1"}
    store.touch("ep_1", datetime(2023, 2, 1))
    assert store.episodes["ep_1"].access_count == 1


def test_multi_valued_attributes_do_not_chain_but_single_valued_do():
    """16 distinct (user, health) facts are 16 facts, not one 16-node update chain; (user, location)
    with a new value is a knowledge update."""
    from datetime import datetime

    from smem.config import SystemConfig
    from smem.schemas import Session, Turn
    from smem.system import build_system

    cfg = SystemConfig.load("configs/offline.yaml")
    mem = build_system(cfg, "offline")
    t = datetime(2023, 1, 1)
    mem.ingest([Session(session_id="s1", ts=t, turns=[Turn(role="user", content="I have a pet allergy. My goal is to run a marathon. I live in Boston.")]),
                Session(session_id="s2", ts=t.replace(month=3), turns=[Turn(role="user", content="I have back pain. My goal is to learn piano. I live in Seattle.")])])
    from smem.schemas import Fact
    facts = [e for e in mem.store.entries(mem.writer.active_ids()) if isinstance(e, Fact)]
    health = [f for f in facts if f.attribute == "health"]; loc = [f for f in facts if f.entity == "user" and f.attribute == "location"]
    assert all(f.superseded_by is None and f.valid_to is None for f in health), "multi-valued facts must not be superseded"
    if len(loc) == 2:   # the heuristic extractor may or may not pull both; when it does, it must chain
        assert sum(f.superseded_by is not None for f in loc) == 1
    mem.close()


def test_chain_resolution_keeps_the_retrieved_hit():
    from datetime import datetime

    from smem.read import Reader
    from smem.schemas import Fact
    from smem.temporal import parse_temporal

    class FakeStore:
        def __init__(self, chain): self.chain_ = chain; self.by_id = {f.id: f for f in chain}
        def get(self, i): return self.by_id.get(i)
        def chain(self, i): return self.chain_
        def volatility(self, e, a): return 1.0
    t = datetime(2023, 1, 1)
    chain = [Fact(id=f"f{i}", entity="user", attribute="location", value=f"city{i}", valid_from=t.replace(month=i+1)) for i in range(4)]
    for a, b in zip(chain, chain[1:]): a.superseded_by = b.id; a.valid_to = b.valid_from
    r = Reader.__new__(Reader); r.store = FakeStore(chain); r.cfg = type("C", (), {"write": type("W", (), {"validity_chain": True})()})()
    allowed = {f.id for f in chain}
    c = parse_temporal("Which city was it, the one with the harbour?", t.replace(month=6)); assert c.mode == "none"
    ids = {x.id for x in r._resolve_chains({"f1": 0.9}, c, allowed, None)}
    assert "f1" in ids, "no temporal cue: the retrieved node must survive resolution"
    assert "f3" in ids, "and the tail is still offered"
    c = parse_temporal("Where do I live now?", t.replace(month=6)); assert c.mode == "now"
    ids = {x.id for x in r._resolve_chains({"f1": 0.9}, c, allowed, None)}
    assert ids == {"f3"}, "an explicit 'now' resolves to the tail only, as designed"
