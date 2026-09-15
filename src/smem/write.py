"""Write policy (plan section 06): deterministic dedup / update first, then streaming submodular
selection under the storage budget B.

`sieve` keeps a geometric grid of admission thresholds, each with its own candidate store
(SieveStreaming); the store with the largest coverage value is the active one. `all` and
`novelty_threshold` are the single-store controls. Physical entries live once in `MemoryStore`
(the pool) and are reference-counted across sieves."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from smem.config import SystemConfig
from smem.coverage import CoverageState
from smem.evict import Evictor
from smem.extract import CANONICAL_ATTRIBUTES
from smem.hawkes import HawkesIntensity, hawkes_keys, source_weight, specificity
from smem.schemas import Entry, ExtractionResult, Fact
from smem.store import MemoryStore


@dataclass
class Sieve:
    threshold: float          # admission threshold as a fraction of the running max gain/token
    cov: CoverageState
    selected: set[str] = field(default_factory=set)
    used: int = 0

    def value(self) -> float:
        return self.cov.value()


@dataclass
class WriteStats:
    candidates: int = 0
    admitted: int = 0
    merged: int = 0
    updated: int = 0
    skipped_threshold: int = 0
    skipped_novelty: int = 0
    skipped_full: int = 0
    swapped_in: int = 0
    evicted: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def sieve_thresholds(cfg: SystemConfig, typical_cost: int | None = None) -> list[float]:
    """Geometric grid of admission thresholds, as fractions of the running max singleton gain per
    token. The low end follows the SieveStreaming guess OPT ≈ m · B / cost (so the threshold is
    m / (2B) per token); the high end is 1/2. `sieve_min_threshold` = 0 means that automatic low end."""
    w = cfg.write
    if typical_cost is None:
        typical_cost = w.sieve_cost_floor
    if w.policy != "sieve":
        return [0.0]
    lo = w.sieve_min_threshold or max(1e-4, typical_cost / (2.0 * max(cfg.budget.store_tokens, 1)))
    lo = min(lo, w.sieve_max_threshold)
    out = []
    t = lo
    while t <= w.sieve_max_threshold + 1e-9:
        out.append(t)
        t *= 1.0 + w.sieve_eps
    return out or [lo]


class WritePolicy:
    def __init__(self, cfg: SystemConfig, store: MemoryStore, evictor: Evictor, hawkes: HawkesIntensity, dim: int):
        self.cfg = cfg
        self.store = store
        self.evictor = evictor
        self.hawkes = hawkes
        self.dim = dim
        self.sieves = [Sieve(t, CoverageState(dim)) for t in sieve_thresholds(cfg)]
        self.stats = WriteStats()
        self.max_ratio = 1e-9              # running max singleton gain per token (sieve normaliser)
        self.refs: Counter[str] = Counter()
        self.candidates: dict[str, Entry] = {}   # every candidate that reached selection, with embedding
        self.written_ids: set[str] = set()        # ever admitted by at least one sieve
        self.evicted_log: list[tuple[str, str]] = []  # (id, reason) for the demo

    # ---- active store --------------------------------------------------------------------------
    def active(self) -> Sieve:
        return max(self.sieves, key=lambda s: (s.value(), -s.threshold))

    def active_ids(self) -> set[str]:
        return set(self.active().selected)

    @property
    def budget(self) -> int:
        return self.cfg.budget.store_tokens

    # ---- weights -------------------------------------------------------------------------------
    def _weight(self, e: Entry, intensities: dict[str, float], now: datetime) -> float:
        lam = self.evictor.hawkes_weight(hawkes_keys(e), now, intensities)
        return specificity(e.text) * source_weight(e.speaker, getattr(e, "kind", None)) * lam

    def _intensities(self, now: datetime) -> dict[str, float]:
        return {e: self.hawkes.intensity(e, now) for e in self.hawkes.entities()}

    def refresh_weights(self, now: datetime) -> None:
        intensities = self._intensities(now)
        weights = {i: self._weight(e, intensities, now) for i, e in self.candidates.items()}
        for s in self.sieves:
            s.cov.set_weights(weights)
        if self.sieves[0].cov.targets:
            self.max_ratio = max(self.sieves[0].cov.max_singleton_ratio(cost_floor=self.cfg.write.sieve_cost_floor), 1e-9)

    # ---- ingestion -----------------------------------------------------------------------------
    def ingest(self, result: ExtractionResult, vecs: dict[str, np.ndarray], now: datetime) -> list[str]:
        """Consider every extracted candidate. `vecs` maps candidate id -> embedding. Returns the ids
        admitted to the active sieve this session."""
        # one mention event per (key, session), whatever the number of candidates that carry it
        mentioned: set[str] = set()
        for e in result.episodes + result.facts:
            mentioned.update(hawkes_keys(e))
        self.hawkes.observe_many(sorted(mentioned), now)
        self.refresh_weights(now)
        intensities = self._intensities(now)

        admitted: list[str] = []
        seen: set[str] = set()
        for ep in result.episodes:
            if ep.id in seen or ep.id in self.candidates:
                continue
            seen.add(ep.id)
            vec = vecs[ep.id]
            ep.embedding = vec.tolist()
            if not ep.raw and self._is_duplicate_episode(vec):
                self.stats.merged += 1
                continue
            self._select(ep, vec, intensities, now)
            if ep.id in self.active().selected:
                admitted.append(ep.id)

        for f in result.facts:
            if f.id in seen or f.id in self.candidates:
                continue
            seen.add(f.id)
            vec = vecs[f.id]
            f.embedding = vec.tolist()
            action, old = self._dedup_fact(f)
            if action == "merge":
                self.stats.merged += 1
                continue
            protect = {old.id} if old is not None else set()
            admitted_anywhere = self._select(f, vec, intensities, now, protect=protect)
            if admitted_anywhere and action == "update" and old is not None:
                self._apply_update(old, f, now)
            if f.id in self.active().selected:
                admitted.append(f.id)
        return admitted

    def _is_duplicate_episode(self, vec: np.ndarray) -> bool:
        pool_eps = set(self.store.episodes)
        if not pool_eps:
            return False
        _, sim = self.store.vectors.max_sim(vec, pool_eps)
        return sim >= self.cfg.write.episode_dup_sim

    def _dedup_fact(self, f: Fact) -> tuple[str, Fact | None]:
        chain = self.store.facts_for(f.entity, f.attribute)
        if not chain:
            return "new", None
        for existing in chain:
            if existing.value.strip().lower() == f.value.strip().lower():
                existing.sources = list(dict.fromkeys(existing.sources + f.sources))
                self.store.update(existing)
                return "merge", existing
        if f.attribute in self.cfg.write.multi_valued_attributes or f.attribute not in CANONICAL_ATTRIBUTES:
            # custom_attribute keys (outside the closed list) are open-ended too: no supersession
            return "new", None
        tail = chain[-1]
        return "update", tail

    def _apply_update(self, old: Fact, new: Fact, now: datetime) -> None:
        self.stats.updated += 1
        if self.cfg.write.validity_chain:
            self.store.supersede(old, new, at=max(new.valid_from, old.valid_from))
        else:
            # overwrite ablation: the old version is dropped everywhere
            for s in self.sieves:
                if old.id in s.selected:
                    self._remove_from(s, old.id, reason="overwrite")

    # ---- streaming selection -------------------------------------------------------------------
    def _select(self, e: Entry, vec: np.ndarray, intensities: dict[str, float], now: datetime,
                protect: set[str] = frozenset()) -> bool:
        self.stats.candidates += 1
        self.candidates[e.id] = e
        w = self._weight(e, intensities, now)
        for s in self.sieves:
            s.cov.add_target(e.id, vec, w, cost=e.tokens)
        singleton = self.sieves[0].cov.singleton(vec) / max(e.tokens, 1)
        # scale estimate only: a very short entry must not set the bar for everyone
        singleton_scale = self.sieves[0].cov.singleton(vec) / max(e.tokens, 1)
        self.max_ratio = max(self.max_ratio, singleton_scale)
        admitted_anywhere = False
        for s in self.sieves:
            if self._consider(s, e, vec, now, protect):
                admitted_anywhere = True
        if admitted_anywhere:
            self.written_ids.add(e.id)
        return admitted_anywhere

    def _consider(self, s: Sieve, e: Entry, vec: np.ndarray, now: datetime, protect: set[str]) -> bool:
        cost = max(e.tokens, 1)
        # A verbatim turn is the source record, not a claim: it is not filtered by the density threshold
        # (its marginal gain over its own paraphrases is always small per token) but it pays its full cost
        # and is swapped or evicted like any entry once the budget binds.
        policy = "fill" if getattr(e, "raw", False) else self.cfg.write.policy
        if policy == "novelty_threshold":
            _, sim = self.store.vectors.max_sim(vec, s.selected) if s.selected else (None, 0.0)
            if 1.0 - sim < self.cfg.write.novelty_threshold:
                self.stats.skipped_novelty += 1
                return False
        elif policy == "sieve":
            gain_ratio = s.cov.gain(vec) / cost
            if gain_ratio < s.threshold * self.max_ratio:
                self.stats.skipped_threshold += 1
                return False

        if s.used + cost <= self.budget:
            self._add_to(s, e, vec)
            return True

        if self.evictor.is_swap:
            victims, loss = self.evictor.swap_victims(s.cov, s.selected, s.used, cost, self.budget, protect)
            if not victims:
                self.stats.skipped_full += 1
                return False
            victim_ids = {m for u in victims for m in u.ids}
            victim_cost = sum(u.cost for u in victims)
            gain_ratio = s.cov.gain_without(vec, victim_ids) / cost
            if gain_ratio >= (1.0 + self.cfg.write.hysteresis_gamma) * (loss / victim_cost):
                for vid in victim_ids:
                    self._remove_from(s, vid, reason="swap")
                self._add_to(s, e, vec)
                self.stats.swapped_in += 1
                return True
            self.stats.skipped_full += 1
            return False

        victims = self.evictor.choose_for_room(s.selected, s.used, cost, self.budget, now, protect)
        if not victims and s.used + cost > self.budget:
            self.stats.skipped_full += 1
            return False
        for u in victims:
            for vid in u.ids:
                self._remove_from(s, vid, reason=self.cfg.evict.policy)
        self._add_to(s, e, vec)
        return True

    def _add_to(self, s: Sieve, e: Entry, vec: np.ndarray) -> None:
        if e.id not in self.store:
            self.store.add(e)
        if e.id in s.selected:
            return
        self.refs[e.id] += 1
        s.selected.add(e.id)
        s.cov.add(e.id, vec)
        s.used += max(e.tokens, 1)
        self.stats.admitted += 1

    def _remove_from(self, s: Sieve, id_: str, reason: str) -> None:
        if id_ not in s.selected:
            return
        entry = self.store.get(id_)
        s.selected.discard(id_)
        s.cov.remove(id_)
        s.used -= max(entry.tokens, 1) if entry is not None else 0
        self.refs[id_] -= 1
        self.stats.evicted += 1
        self.evicted_log.append((id_, reason))
        if self.refs[id_] <= 0:
            del self.refs[id_]
            self.store.remove(id_)

    # ---- external admission (consolidation summaries) ------------------------------------------
    def admit_external(self, e: Entry, vec: np.ndarray, now: datetime) -> bool:
        """Admission for entries produced inside the system (consolidation summaries)."""
        e.embedding = vec.tolist()
        return self._select(e, vec, self._intensities(now), now)

    def drop(self, ids: set[str], reason: str = "consolidated") -> None:
        for s in self.sieves:
            for i in list(ids):
                self._remove_from(s, i, reason)


