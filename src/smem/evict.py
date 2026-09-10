"""Eviction (plan section 06). Two families behind one class:

  * room-making controls (fifo / lru / random / utility_heuristic): rank units, evict the lowest
    until the new candidate fits; the candidate is always admitted.
  * swap (swap / swap_no_hawkes): the streaming-submodular rule; the victim is the unit with the
    smallest coverage loss per token, and the candidate replaces it only if its gain per token beats
    the victim's by the hysteresis factor (1 + γ). Otherwise the candidate is skipped.

A validity chain is one unit: scored as a whole, evicted as a whole. Consolidated episodes are
considered before anything else."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime

from smem.config import SystemConfig
from smem.coverage import CoverageState
from smem.hawkes import HawkesIntensity, source_weight, specificity
from smem.schemas import Episode, Fact
from smem.store import MemoryStore


@dataclass(frozen=True)
class Unit:
    ids: tuple[str, ...]
    cost: int
    consolidated: bool


class Evictor:
    def __init__(self, cfg: SystemConfig, store: MemoryStore, hawkes: HawkesIntensity):
        self.cfg = cfg
        self.policy = cfg.evict.policy
        self.store = store
        self.hawkes = hawkes
        self.rng = random.Random(cfg.evict.seed)
        self.weights = cfg.evict.heuristic_weights

    @property
    def is_swap(self) -> bool:
        return self.policy in ("swap", "swap_no_hawkes")

    def units(self, selected: set[str]) -> list[Unit]:
        seen: set[str] = set()
        out: list[Unit] = []
        for id_ in selected:
            if id_ in seen:
                continue
            entry = self.store.get(id_)
            if entry is None:
                continue
            if isinstance(entry, Fact):
                members = [f.id for f in self.store.chain(id_) if f.id in selected] or [id_]
            else:
                members = [id_]
            seen.update(members)
            entries = [self.store.get(m) for m in members]
            cost = sum(e.tokens for e in entries if e is not None)
            consolidated = all(isinstance(e, Episode) and e.consolidated for e in entries)
            out.append(Unit(tuple(members), max(cost, 1), consolidated))
        return out

    # ---- room-making controls ------------------------------------------------------------------
    def _rank_key(self, unit: Unit, now: datetime) -> float:
        entries = [self.store.get(m) for m in unit.ids]
        entries = [e for e in entries if e is not None]
        if self.policy == "fifo":
            return float(min(self.store.insert_order(m) for m in unit.ids))
        if self.policy == "lru":
            return min((e.last_access or e.ts).timestamp() for e in entries)
        if self.policy == "random":
            return self.rng.random()
        if self.policy == "utility_heuristic":
            return self._utility(entries, now)
        raise ValueError(f"not a ranking policy: {self.policy}")

    def _utility(self, entries: list, now: datetime) -> float:
        # the v1 weighted-sum heuristic: recency, frequency, specificity, source
        last = max((e.last_access or e.ts) for e in entries)
        age_days = max((now - last).total_seconds() / 86400.0, 0.0)
        recency = math.exp(-age_days / 30.0)
        frequency = math.log1p(sum(e.access_count for e in entries)) / math.log1p(50)
        spec = max(specificity(e.text) for e in entries)
        src = max(source_weight(e.speaker, getattr(e, "kind", None)) for e in entries)
        w = self.weights
        return (w.get("recency", 0) * recency + w.get("frequency", 0) * min(frequency, 1.0)
                + w.get("specificity", 0) * spec + w.get("source", 0) * src)

    def choose_for_room(self, selected: set[str], used: int, needed: int, budget: int,
                        now: datetime, protect: set[str] = frozenset()) -> list[Unit]:
        """Evict the lowest-ranked units until used - freed + needed ≤ budget."""
        units = [u for u in self.units(selected) if not set(u.ids) & protect]
        units.sort(key=lambda u: (not u.consolidated, self._rank_key(u, now)))
        victims: list[Unit] = []
        freed = 0
        for u in units:
            if used - freed + needed <= budget:
                break
            victims.append(u)
            freed += u.cost
        if used - freed + needed > budget:
            return []  # cannot make room (candidate larger than the whole store)
        return victims

    # ---- swap rule -----------------------------------------------------------------------------
    def swap_victims(self, cov: CoverageState, selected: set[str], used: int, needed: int, budget: int,
                     protect: set[str] = frozenset()) -> tuple[list[Unit], float]:
        """Pick units with the smallest loss per token until the candidate would fit. Returns the
        victims and their total loss (Δf of the whole victim set)."""
        units = [u for u in self.units(selected) if not set(u.ids) & protect]
        if not units:
            return [], 0.0
        single_losses = cov.losses()
        scored: list[tuple[float, Unit]] = []
        for u in units:
            if len(u.ids) == 1:
                loss = single_losses.get(u.ids[0], 0.0)
            else:
                loss = cov.loss_of_set(set(u.ids))
            scored.append((loss / u.cost, u))
        # consolidated episodes are tried first; among each tier, smallest loss per token first
        scored.sort(key=lambda p: (not p[1].consolidated, p[0]))
        victims: list[Unit] = []
        freed = 0
        for _, u in scored:
            if used - freed + needed <= budget:
                break
            victims.append(u)
            freed += u.cost
        if used - freed + needed > budget:
            return [], 0.0
        if len(victims) == 1:
            total_loss = scored[[u for _, u in scored].index(victims[0])][0] * victims[0].cost
        else:
            total_loss = cov.loss_of_set({m for u in victims for m in u.ids})
        return victims, total_loss

    def hawkes_weight(self, entities: list[str], now: datetime, intensities: dict[str, float] | None = None) -> float:
        if self.policy == "swap_no_hawkes":
            return 1.0
        if intensities is not None:
            vals = [intensities.get(e.lower(), self.hawkes.mu_floor) for e in entities]
            return max(vals) if vals else self.hawkes.mu_floor
        return self.hawkes.max_intensity([e.lower() for e in entities], now)
