import numpy as np

from smem.config import SystemConfig
from smem.consolidate import LexicalNLI, threshold_clusters
from smem.system import SelectiveMemory
from tests.conftest import session


def near_duplicate_sessions(n=5):
    return [session(f"r{i}", i, f"I cooked the garlic tomato pasta for dinner again and loved it, night {i}.") for i in range(n)]


def test_threshold_clusters_groups_similar_vectors():
    vecs = np.array([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99], [0.7, 0.7]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    groups = threshold_clusters(vecs, 0.95)
    assert sorted(sorted(g) for g in groups) == [[0, 1], [2, 3], [4]]


def test_redundancy_trigger_fires_and_marks_members():
    cfg = SystemConfig.load("configs/offline.yaml", {"write.policy": "all", "write.episode_dup_sim": 1.01,
                                                       "consolidation.redundancy_threshold": 0.9,
                                                       "consolidation.nli_threshold": 0.3})
    mem = SelectiveMemory(cfg)
    mem.ingest(near_duplicate_sessions())
    st = mem.consolidator.stats
    assert st.clusters_fired >= 1 and st.summaries_accepted >= 1 and st.episodes_consolidated >= 3
    assert any(e.consolidated for e in mem.store.episodes.values())
    assert mem.summary_sources()
    summary_ids = set(mem.summary_sources())
    assert summary_ids & mem.surviving_ids()


def test_nli_rejection_leaves_cluster_untouched():
    class RejectAll:
        def entails(self, premise, hypothesis):
            return 0.0

    cfg = SystemConfig.load("configs/offline.yaml", {"write.policy": "all", "write.episode_dup_sim": 1.01,
                                                       "consolidation.redundancy_threshold": 0.9})
    mem = SelectiveMemory(cfg, nli=RejectAll())
    mem.ingest(near_duplicate_sessions())
    st = mem.consolidator.stats
    assert st.clusters_fired >= 1 and st.summaries_accepted == 0 and st.summaries_rejected >= 1
    assert not any(e.consolidated for e in mem.store.episodes.values())


def test_no_consolidation_and_fixed_interval_switches():
    base = {"write.policy": "all", "write.episode_dup_sim": 1.01}
    off = SelectiveMemory(SystemConfig.load("configs/offline.yaml", {**base, "consolidation.policy": "no_consolidation"}))
    off.ingest(near_duplicate_sessions())
    assert off.consolidator.stats.runs == 0
    fixed = SelectiveMemory(SystemConfig.load("configs/offline.yaml", {**base, "consolidation.policy": "fixed_interval",
                                                                       "consolidation.fixed_interval": 5,
                                                                       "consolidation.nli_threshold": 0.0}))
    fixed.ingest(near_duplicate_sessions(5))
    assert fixed.consolidator.stats.runs == 1 and fixed.consolidator.stats.clusters_fired >= 1


def test_lexical_nli():
    nli = LexicalNLI()
    assert nli.entails("user cooked garlic tomato pasta many nights", "I cooked garlic tomato pasta") > 0.6
    assert nli.entails("user likes running", "I cooked garlic tomato pasta") < 0.3


def test_consolidated_episodes_are_first_out_under_pressure():
    cfg = SystemConfig.load("configs/offline.yaml", {"write.policy": "all", "evict.policy": "swap", "write.episode_dup_sim": 1.01,
                                                       "consolidation.redundancy_threshold": 0.9, "consolidation.nli_threshold": 0.3,
                                                       "budget.store_tokens": 130, "write.hysteresis_gamma": 0.0})
    mem = SelectiveMemory(cfg)
    mem.ingest(near_duplicate_sessions())
    consolidated_before = {i for i, e in mem.store.episodes.items() if e.consolidated}
    assert consolidated_before
    for i in range(6):
        topic = ["work", "travel", "music", "sport", "art", "tax"][i]
        mem.ingest_session(session(f"n{i}", 10 + i, f"I went to a new unrelated event about {topic} number {i}."))
    survivors = mem.surviving_ids()
    assert len(consolidated_before & survivors) < len(consolidated_before)


def test_summary_supported_direction_keeps_grounded_facts_and_drops_hallucinated_ones():
    """New default gate: a summary fact survives only if some member entails it. The old rule asked the
    summary to entail every member, which a lossy summary cannot do (229/230 rejected on dev)."""
    from smem.consolidate import LexicalNLI, _summary_fact

    class Scripted:
        def summarize(self, members, vecs):
            return [_summary_fact("user", "favorite", "garlic tomato pasta", members),   # words in the members
                    _summary_fact("user", "location", "Zanzibar volcano lodge", members)]   # invented

    cfg = SystemConfig.load("configs/offline.yaml", {"write.policy": "all", "write.episode_dup_sim": 1.01,
                                                       "consolidation.redundancy_threshold": 0.9,
                                                       "consolidation.nli_threshold": 0.5})
    mem = SelectiveMemory(cfg, summarizer=Scripted(), nli=LexicalNLI())
    mem.ingest(near_duplicate_sessions())
    st = mem.consolidator.stats
    assert st.clusters_fired >= 1 and st.summaries_accepted >= 1
    admitted = [f for f in mem.store.facts.values() if f.id.startswith("fs_")]
    assert admitted and all("Zanzibar" not in f.value for f in admitted)       # hallucinated fact dropped
    assert any(f.value == "garlic tomato pasta" for f in admitted)              # grounded fact kept

    old = SystemConfig.load("configs/offline.yaml", {"write.policy": "all", "write.episode_dup_sim": 1.01,
                                                       "consolidation.redundancy_threshold": 0.9,
                                                       "consolidation.nli_direction": "members_entailed"})
    mem2 = SelectiveMemory(old, summarizer=Scripted(), nli=LexicalNLI())
    mem2.ingest(near_duplicate_sessions())
    assert mem2.consolidator.stats.summaries_accepted == 0                      # the original rule rejects
