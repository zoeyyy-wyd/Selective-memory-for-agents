"""The one objective (plan section 05).

    f_W(S) = Σ_{h ∈ H} w_h · max_{s ∈ S} sim(h, s)      (facility location; monotone submodular)

`CoverageState` maintains f incrementally. For every target h it keeps the best and second-best
similarity over the selected set S and which member provides each. Adding x touches only the targets
whose top-2 changes; removing y recomputes only the targets y was covering. Members of S live in
slots of one matrix so that per-member losses come out of a single bincount.

The same class serves the storage scale (H = all candidates seen so far, S = the store, budget B) and
the read scale (H = retrieval candidates, S = the packed context, budget R)."""

from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np


class CoverageState:
    def __init__(self, dim: int):
        self.dim = dim
        # targets H
        self._h_ids: list[str] = []
        self._h_index: dict[str, int] = {}
        self._h_vecs = np.zeros((0, dim), dtype=np.float32)
        self._w = np.zeros(0, dtype=np.float32)
        self._cost = np.ones(0, dtype=np.float32)
        self._best = np.zeros(0, dtype=np.float32)
        self._second = np.zeros(0, dtype=np.float32)
        self._best_slot = np.zeros(0, dtype=np.int64)    # -1 = none
        self._second_slot = np.zeros(0, dtype=np.int64)
        # selected S, in slots
        self._s_mat = np.zeros((0, dim), dtype=np.float32)
        self._slot_of: dict[str, int] = {}
        self._id_of_slot: list[str | None] = []
        self._free: list[int] = []

    # ---- targets -------------------------------------------------------------------------------
    def add_target(self, h_id: str, vec: np.ndarray, weight: float, cost: int = 1) -> None:
        if h_id in self._h_index:
            self.set_weight(h_id, weight)
            return
        vec = np.asarray(vec, dtype=np.float32)
        self._h_index[h_id] = len(self._h_ids)
        self._h_ids.append(h_id)
        self._h_vecs = np.vstack([self._h_vecs, vec[None, :]])
        self._w = np.append(self._w, np.float32(weight))
        self._cost = np.append(self._cost, np.float32(max(cost, 1)))
        self._best = np.append(self._best, np.float32(0.0))
        self._second = np.append(self._second, np.float32(0.0))
        self._best_slot = np.append(self._best_slot, -1)
        self._second_slot = np.append(self._second_slot, -1)
        self._recompute_rows(np.array([len(self._h_ids) - 1]))

    def set_weight(self, h_id: str, weight: float) -> None:
        self._w[self._h_index[h_id]] = weight

    def set_weights(self, weights: dict[str, float]) -> None:
        for h_id, w in weights.items():
            idx = self._h_index.get(h_id)
            if idx is not None:
                self._w[idx] = w

    def weight(self, h_id: str) -> float:
        return float(self._w[self._h_index[h_id]])

    def has_target(self, h_id: str) -> bool:
        return h_id in self._h_index

    @property
    def targets(self) -> list[str]:
        return list(self._h_ids)

    # ---- selected set --------------------------------------------------------------------------
    @property
    def selected(self) -> list[str]:
        return [i for i in self._id_of_slot if i is not None]

    def __contains__(self, s_id: str) -> bool:
        return s_id in self._slot_of

    def __len__(self) -> int:
        return len(self._slot_of)

    def _active_slots(self) -> np.ndarray:
        return np.array(sorted(self._slot_of.values()), dtype=np.int64)

    def value(self) -> float:
        return float(np.dot(self._w, self._best))

    def max_singleton_ratio(self) -> float:
        """max_h f({h}) / cost(h) over all targets under the current weights: the SieveStreaming
        normaliser. One |H|×|H| product; H is in the low thousands."""
        n = len(self._h_ids)
        if n == 0:
            return 0.0
        sims = np.maximum(self._h_vecs @ self._h_vecs.T, 0.0)
        singles = sims @ self._w
        return float(np.max(singles / self._cost))

    def singleton(self, vec: np.ndarray) -> float:
        """f({x}): coverage x would provide on its own."""
        if len(self._h_ids) == 0:
            return 0.0
        sims = self._h_vecs @ np.asarray(vec, dtype=np.float32)
        return float(np.dot(self._w, np.maximum(sims, 0.0)))

    def gain(self, vec: np.ndarray) -> float:
        """Δf(x | S) for a candidate x with embedding vec (x need not be a target)."""
        if len(self._h_ids) == 0:
            return 0.0
        sims = self._h_vecs @ np.asarray(vec, dtype=np.float32)
        return float(np.dot(self._w, np.maximum(sims - self._best, 0.0)))

    def gain_without(self, vec: np.ndarray, removed: set[str]) -> float:
        """Δf(x | S ∖ removed): the gain x would have if `removed` were evicted first."""
        if len(self._h_ids) == 0:
            return 0.0
        base = self._best.copy()
        slots = [self._slot_of[r] for r in removed if r in self._slot_of]
        if slots:
            rows = np.flatnonzero(np.isin(self._best_slot, slots))
            if len(rows):
                base[rows] = self._max_excluding(rows, set(slots))
        sims = self._h_vecs @ np.asarray(vec, dtype=np.float32)
        return float(np.dot(self._w, np.maximum(sims - base, 0.0)))

    def losses(self) -> dict[str, float]:
        """Δf(y | S ∖ y) for every y ∈ S at once."""
        if not self._slot_of:
            return {}
        n_slots = len(self._id_of_slot)
        mask = self._best_slot >= 0
        contrib = self._w[mask] * (self._best[mask] - self._second[mask])
        per_slot = np.bincount(self._best_slot[mask], weights=contrib, minlength=n_slots)
        return {s_id: float(per_slot[slot]) for s_id, slot in self._slot_of.items()}

    def loss(self, s_id: str) -> float:
        slot = self._slot_of.get(s_id)
        if slot is None:
            return 0.0
        mask = self._best_slot == slot
        return float(np.dot(self._w[mask], self._best[mask] - self._second[mask]))

    def loss_of_set(self, s_ids: set[str]) -> float:
        """Δf(Y | S ∖ Y) for a group evicted as one unit (a validity chain)."""
        slots = {self._slot_of[i] for i in s_ids if i in self._slot_of}
        if not slots:
            return 0.0
        if len(slots) == 1:
            return self.loss(next(i for i in s_ids if i in self._slot_of))
        rows = np.flatnonzero(np.isin(self._best_slot, list(slots)))
        if len(rows) == 0:
            return 0.0
        alt = self._max_excluding(rows, slots)
        return float(np.dot(self._w[rows], self._best[rows] - alt))

    def add(self, s_id: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32)
        if s_id in self._slot_of:
            return
        if self._free:
            slot = self._free.pop()
            self._s_mat[slot] = vec
            self._id_of_slot[slot] = s_id
        else:
            slot = len(self._id_of_slot)
            self._s_mat = np.vstack([self._s_mat, vec[None, :]])
            self._id_of_slot.append(s_id)
        self._slot_of[s_id] = slot
        if len(self._h_ids) == 0:
            return
        sims = np.maximum(self._h_vecs @ vec, 0.0)
        beats_best = sims > self._best
        beats_second = (~beats_best) & (sims > self._second)
        self._second[beats_best] = self._best[beats_best]
        self._second_slot[beats_best] = self._best_slot[beats_best]
        self._best[beats_best] = sims[beats_best]
        self._best_slot[beats_best] = slot
        self._second[beats_second] = sims[beats_second]
        self._second_slot[beats_second] = slot

    def remove(self, s_id: str) -> None:
        slot = self._slot_of.pop(s_id, None)
        if slot is None:
            return
        self._id_of_slot[slot] = None
        self._s_mat[slot] = 0.0
        self._free.append(slot)
        rows = np.flatnonzero((self._best_slot == slot) | (self._second_slot == slot))
        if len(rows):
            self._recompute_rows(rows)

    def _recompute_rows(self, rows: np.ndarray) -> None:
        active = self._active_slots()
        if len(active) == 0:
            self._best[rows] = 0.0
            self._second[rows] = 0.0
            self._best_slot[rows] = -1
            self._second_slot[rows] = -1
            return
        sims = np.maximum(self._h_vecs[rows] @ self._s_mat[active].T, 0.0)  # |rows| × |active|
        if len(active) == 1:
            self._best[rows] = sims[:, 0]
            self._best_slot[rows] = np.where(sims[:, 0] > 0, active[0], -1)
            self._second[rows] = 0.0
            self._second_slot[rows] = -1
            return
        top2 = np.argpartition(-sims, 1, axis=1)[:, :2]
        first = np.take_along_axis(sims, top2[:, :1], axis=1)[:, 0]
        second = np.take_along_axis(sims, top2[:, 1:2], axis=1)[:, 0]
        swap = second > first
        first, second = np.where(swap, second, first), np.where(swap, first, second)
        first_slot = np.where(swap, top2[:, 1], top2[:, 0])
        second_slot = np.where(swap, top2[:, 0], top2[:, 1])
        self._best[rows] = first
        self._second[rows] = second
        self._best_slot[rows] = np.where(first > 0, active[first_slot], -1)
        self._second_slot[rows] = np.where(second > 0, active[second_slot], -1)

    def _max_excluding(self, rows: np.ndarray, excluded: set[int]) -> np.ndarray:
        active = np.array([s for s in sorted(self._slot_of.values()) if s not in excluded], dtype=np.int64)
        if len(active) == 0:
            return np.zeros(len(rows), dtype=np.float32)
        sims = np.maximum(self._h_vecs[rows] @ self._s_mat[active].T, 0.0)
        return sims.max(axis=1)


def budgeted_greedy(
    items: list[str],
    cost: dict[str, int],
    gain_fn: Callable[[str, list[str]], float],
    budget: int,
) -> list[str]:
    """Khuller–Moss–Naor budgeted maximum coverage: greedy on gain/cost with CELF lazy evaluation,
    then the better of the greedy set and the best single affordable item ((1 − 1/e)/2 guarantee).
    gain_fn(item, selected) returns the marginal gain of item given the selected list and must be
    submodular for the lazy bound to hold."""
    affordable = [i for i in items if cost[i] <= budget]
    singles = {i: gain_fn(i, []) for i in affordable}
    selected: list[str] = []
    used = 0
    greedy_value = 0.0
    heap = [(-singles[i] / max(cost[i], 1), i, 0) for i in affordable]
    heapq.heapify(heap)
    round_no = 0
    while heap:
        neg_ratio, item, stamp = heapq.heappop(heap)
        if used + cost[item] > budget:
            continue
        if stamp != round_no:
            fresh = gain_fn(item, selected) / max(cost[item], 1)
            heapq.heappush(heap, (-fresh, item, round_no))
            continue
        if -neg_ratio <= 0:
            break
        greedy_value += -neg_ratio * max(cost[item], 1)
        selected.append(item)
        used += cost[item]
        round_no += 1

    best_single = max(affordable, key=lambda i: singles[i], default=None)
    if best_single is not None and singles[best_single] > greedy_value:
        return [best_single]
    return selected
