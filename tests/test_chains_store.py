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
