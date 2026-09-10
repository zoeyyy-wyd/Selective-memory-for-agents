"""Predicted access intensity as a univariate Hawkes process per entity (plan section 05):

    λ_e(t) = μ_e + Σ_{t_i < t} α · exp(−β (t − t_i))

Events are mentions of the entity in the history and retrieval hits at read time. μ_e is the
entity's mention rate over the history. α, β are shared across entities and fit on dev by grid
search over the log-likelihood. Time is measured in days."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime


def _days(t: datetime, origin: datetime) -> float:
    return (t - origin).total_seconds() / 86400.0


class HawkesIntensity:
    def __init__(self, alpha: float = 0.5, beta: float = 0.1, mu_floor: float = 0.01):
        self.alpha = alpha
        self.beta = beta
        self.mu_floor = mu_floor
        self._events: dict[str, list[float]] = defaultdict(list)
        self._origin: datetime | None = None
        self._t_min: float | None = None
        self._t_max: float | None = None

    def _to_days(self, t: datetime) -> float:
        if self._origin is None:
            self._origin = t
        return _days(t, self._origin)

    def observe(self, entity: str, t: datetime) -> None:
        x = self._to_days(t)
        self._events[entity].append(x)
        self._t_min = x if self._t_min is None else min(self._t_min, x)
        self._t_max = x if self._t_max is None else max(self._t_max, x)

    def observe_many(self, entities: list[str], t: datetime) -> None:
        for e in set(entities):
            self.observe(e, t)

    def base_rate(self, entity: str) -> float:
        events = self._events.get(entity, [])
        if not events or self._t_min is None or self._t_max is None:
            return self.mu_floor
        # rate over the observed history, floored at 30 days so the first sessions do not
        # produce "five mentions per day" estimates
        span = max(self._t_max - self._t_min, 30.0)
        return max(self.mu_floor, len(events) / span)

    def intensity(self, entity: str, t: datetime) -> float:
        """λ_e(t⁺): events at exactly t count, so an entity mentioned in the current session is hot."""
        x = self._to_days(t)
        events = self._events.get(entity, [])
        excitation = sum(self.alpha * math.exp(-self.beta * (x - ti)) for ti in events if ti <= x)
        return self.base_rate(entity) + excitation

    def max_intensity(self, entities: list[str], t: datetime) -> float:
        if not entities:
            return self.mu_floor
        return max(self.intensity(e, t) for e in entities)

    # ---- fitting -------------------------------------------------------------------------------
    def log_likelihood(self, alpha: float, beta: float) -> float:
        """Σ_e [ Σ_i log λ_e(t_i) − ∫_0^T λ_e(t) dt ] with T = the last observed time."""
        if self._t_max is None:
            return 0.0
        T = self._t_max + 1e-6
        total = 0.0
        for entity, events in self._events.items():
            mu = self.base_rate(entity)
            ev = sorted(events)
            # recursive form of the excitation term: A_i = exp(-β Δt) (1 + A_{i-1})
            A = 0.0
            for i, ti in enumerate(ev):
                if i > 0:
                    A = math.exp(-beta * (ti - ev[i - 1])) * (1.0 + A)
                total += math.log(mu + alpha * A)
            compensator = mu * T + sum((alpha / beta) * (1 - math.exp(-beta * (T - ti))) for ti in ev)
            total -= compensator
        return total

    def fit(self, alphas=None, betas=None) -> tuple[float, float]:
        alphas = list(alphas) if alphas is not None else [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
        betas = list(betas) if betas is not None else [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
        best = (-math.inf, self.alpha, self.beta)
        for a in alphas:
            for b in betas:
                ll = self.log_likelihood(a, b)
                if ll > best[0]:
                    best = (ll, a, b)
        _, self.alpha, self.beta = best
        return self.alpha, self.beta

    def entities(self) -> list[str]:
        return list(self._events)


def hawkes_keys(entry) -> list[str]:
    """Which point processes an entry belongs to. Facts: their entity/attribute (e.g. 'user/city') plus
    the value when it is a name; episodes: their named entities. The generic 'user' entity is only used
    when nothing more specific exists, otherwise it would dominate every weight."""
    from smem.schemas import Fact

    if isinstance(entry, Fact):
        keys = [f"{entry.entity.lower()}/{entry.attribute.lower()}"]
        if entry.value[:1].isupper():
            keys.append(entry.value.lower())
        return keys
    keys = [e.lower() for e in entry.entities if e.lower() != "user"]
    return keys or ["user"]


def specificity(text: str) -> float:
    """Cheap specificity prior in [0.2, 1]: entries with numbers, capitalised names or dates are
    more specific than generic chatter. Used as spec(h) in the storage-scale weight."""
    if not text:
        return 0.2
    toks = text.split()
    if not toks:
        return 0.2
    signal = 0
    for tok in toks:
        if any(ch.isdigit() for ch in tok):
            signal += 2
        elif tok[:1].isupper() and len(tok) > 1:
            signal += 1
    ratio = min(1.0, signal / max(len(toks), 1) * 2)
    return float(0.2 + 0.8 * ratio)


def source_weight(speaker: str, kind: str | None = None) -> float:
    """src(h): user statements outrank assistant text; inferred facts rank lowest."""
    if kind == "inferred":
        return 0.5
    if speaker == "user":
        return 1.0
    return 0.6


