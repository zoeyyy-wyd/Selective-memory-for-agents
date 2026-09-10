import itertools

import numpy as np
import pytest

from smem.coverage import CoverageState, budgeted_greedy
from smem.embed import normalize

rng = np.random.default_rng(0)


def brute_force_value(H, w, S_vecs):
    if not S_vecs:
        return 0.0
    S = np.stack(S_vecs)
    return float(sum(w[i] * max(0.0, float(np.max(H[i] @ S.T))) for i in range(len(H))))


def make_state(n_h=12, dim=16):
    H = normalize(rng.normal(size=(n_h, dim)))
    w = rng.uniform(0.2, 1.0, size=n_h)
    cov = CoverageState(dim)
    for i in range(n_h):
        cov.add_target(f"h{i}", H[i], float(w[i]), cost=3)
    return cov, H, w


def test_incremental_value_matches_brute_force_through_adds_and_removes():
    cov, H, w = make_state()
    S = {}
    for step in range(30):
        if S and rng.random() < 0.35:
            k = rng.choice(list(S))
            cov.remove(k)
            del S[k]
        else:
            k = f"s{step}"
            v = normalize(rng.normal(size=(1, 16)))[0]
            expected_gain = brute_force_value(H, w, list(S.values()) + [v]) - brute_force_value(H, w, list(S.values()))
            assert cov.gain(v) == pytest.approx(expected_gain, abs=1e-4)
            cov.add(k, v)
            S[k] = v
        assert cov.value() == pytest.approx(brute_force_value(H, w, list(S.values())), abs=1e-4)


def test_losses_match_value_difference():
    cov, H, w = make_state()
    S = {f"s{i}": normalize(rng.normal(size=(1, 16)))[0] for i in range(6)}
    for k, v in S.items():
        cov.add(k, v)
    losses = cov.losses()
    for k in S:
        others = [v for kk, v in S.items() if kk != k]
        expected = brute_force_value(H, w, list(S.values())) - brute_force_value(H, w, others)
        assert losses[k] == pytest.approx(expected, abs=1e-4)
        assert cov.loss(k) == pytest.approx(expected, abs=1e-4)
    # a set of two, evicted together
    ks = list(S)[:2]
    rest = [v for kk, v in S.items() if kk not in ks]
    expected = brute_force_value(H, w, list(S.values())) - brute_force_value(H, w, rest)
    assert cov.loss_of_set(set(ks)) == pytest.approx(expected, abs=1e-4)
    # gain of a new vector once those two are gone
    v = normalize(rng.normal(size=(1, 16)))[0]
    expected_gain = brute_force_value(H, w, rest + [v]) - brute_force_value(H, w, rest)
    assert cov.gain_without(v, set(ks)) == pytest.approx(expected_gain, abs=1e-4)


def test_budgeted_greedy_respects_budget_and_is_near_optimal():
    items = [f"i{k}" for k in range(7)]
    cost = {"i0": 4, "i1": 3, "i2": 3, "i3": 2, "i4": 5, "i5": 1, "i6": 2}
    # weighted set-cover style gains: each item covers a set of elements
    covers = {"i0": {1, 2, 3}, "i1": {3, 4}, "i2": {5, 6}, "i3": {1}, "i4": {7, 8, 9, 10}, "i5": {2}, "i6": {6, 7}}
    weights = {e: 1.0 for e in range(1, 11)}

    def gain_fn(item, selected):
        covered = set().union(*(covers[s] for s in selected)) if selected else set()
        return sum(weights[e] for e in covers[item] - covered)

    budget = 8
    chosen = budgeted_greedy(items, cost, gain_fn, budget)
    assert sum(cost[c] for c in chosen) <= budget
    best = 0.0
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            if sum(cost[c] for c in combo) <= budget:
                best = max(best, gain_fn(combo[0], []) + sum(gain_fn(c, list(combo[:i])) for i, c in enumerate(combo) if i))
    assert gain_fn(chosen[0], []) + sum(gain_fn(c, chosen[:i]) for i, c in enumerate(chosen) if i) >= 0.5 * (1 - 1 / np.e) * best


def test_budgeted_greedy_prefers_single_big_item_when_better():
    items = ["big", "a", "b"]
    cost = {"big": 5, "a": 1, "b": 1}
    covers = {"big": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "a": {1, 2}, "b": {3}}

    def gain_fn(item, selected):
        covered = set().union(*(covers[s] for s in selected)) if selected else set()
        return len(covers[item] - covered)

    assert budgeted_greedy(items, cost, gain_fn, 5) == ["big"]
