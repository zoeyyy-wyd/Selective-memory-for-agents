"""Read path (plan section 09): temporal constraint -> query rewriting -> hybrid retrieval (BM25 ⊕
dense, RRF) -> validity-chain resolution -> optional entity two-hop -> budgeted packing under R ->
abstention decision. Packing is the read-scale instance of the coverage objective; MMR and top-k are
the controls."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from smem.config import SystemConfig
from smem.coverage import CoverageState, budgeted_greedy
from smem.embed import Embedder, tokenize
from smem.hawkes import HawkesIntensity, hawkes_keys
from smem.llm import LLM
from smem.schemas import Entry, Fact
from smem.store import MemoryStore
from smem.temporal import TemporalConstraint, parse_temporal

_TEMPORAL_WORDS = re.compile(
    r"\b(now|currently|current|today|last|past|previous|ago|recently|in|during|when|did|do|does|the|a|an|my|i|me|"
    r"what|which|where|how|many|much|is|are|was|were|have|has|had|you|remember|tell|about|of|to)\b", re.IGNORECASE)
_LOW_VOLATILITY = 1.0 / 30.0   # fewer than one update a month: treat "now" contradictions as suspect
_STOP_Q = frozenset({"what", "which", "where", "when", "how", "many", "much", "did", "do", "does", "is", "are", "was", "were", "the", "a", "an", "my", "i", "me", "you", "your", "of", "to", "in", "on", "at", "for", "about", "tell", "remember", "have", "has", "had", "that", "this", "it", "and", "or"})


@dataclass
class Candidate:
    id: str
    rel: float
    hop: int = 1
    reason: str = "retrieval"


@dataclass
class ReadResult:
    question: str
    constraint: TemporalConstraint
    queries: list[str]
    candidates: list[Candidate]
    packed: list[Entry]
    packed_rel: dict[str, float]
    tokens: int
    abstain: bool
    top_rel: float
    hop2_budget: int = 0
    diagnostics: dict = field(default_factory=dict)

    @property
    def packed_ids(self) -> set[str]:
        return {e.id for e in self.packed}


REWRITE_PROMPT = """Rewrite the question into one or two short search queries that would match memory entries
written as short first-person facts about the user (e.g. "user city: Seattle", "user moved to Boston").
Return JSON: {"queries": ["...", "..."]}"""


class Reader:
    def __init__(self, cfg: SystemConfig, store: MemoryStore, embedder: Embedder, hawkes: HawkesIntensity,
                 rewriter: LLM | None = None):
        self.cfg = cfg
        self.r = cfg.read
        self.store = store
        self.embedder = embedder
        self.hawkes = hawkes
        self.rewriter = rewriter

    # ---- entry point ---------------------------------------------------------------------------
    def read(self, question: str, now: datetime, allowed: set[str]) -> ReadResult:
        constraint = parse_temporal(question, now)
        queries = self.rewrite(question)
        qvec = self.embedder.encode([question])[0]
        fused = self._retrieve(queries, qvec, allowed)
        cands = self._resolve_chains(fused, constraint, allowed, qvec)
        hop2_budget = 0
        if self.r.retrieval == "two_hop" and cands:
            cands, hop2_budget = self._second_hop(cands, allowed, qvec)
        packed_ids, packed_rel = self._pack(cands, qvec, hop2_budget)
        packed = self.store.entries(packed_ids)
        tokens = sum(e.tokens for e in packed)
        top_rel = max((packed_rel[i] for i in packed_ids), default=0.0)
        abstain = self._should_abstain(question, packed, top_rel)
        hit_keys: set[str] = set()
        for e in packed:
            self.store.touch(e.id, now)
            hit_keys.update(hawkes_keys(e))
        self.hawkes.observe_many(sorted(hit_keys), now)
        return ReadResult(question=question, constraint=constraint, queries=queries, candidates=cands,
                          packed=packed, packed_rel=packed_rel, tokens=tokens, abstain=abstain,
                          top_rel=top_rel, hop2_budget=hop2_budget,
                          diagnostics={"n_candidates": len(cands), "n_hop2": sum(1 for c in cands if c.hop == 2)})

    # ---- rewriting -----------------------------------------------------------------------------
    def rewrite(self, question: str) -> list[str]:
        queries = [question]
        if not self.r.rewrite_queries:
            return queries
        if self.rewriter is not None:
            from smem.llm import parse_json_object

            obj = parse_json_object(self.rewriter.complete(REWRITE_PROMPT, question, max_tokens=120))
            if obj and isinstance(obj.get("queries"), list):
                queries += [str(q) for q in obj["queries"][:2] if str(q).strip()]
                return list(dict.fromkeys(queries))
        stripped = " ".join(w for w in tokenize(question) if w not in _STOP_Q)
        if stripped and stripped != question.lower():
            queries.append("user " + stripped)
        return queries

    # ---- retrieval -----------------------------------------------------------------------------
    def _retrieve(self, queries: list[str], qvec: np.ndarray, allowed: set[str]) -> dict[str, float]:
        rrf: dict[str, float] = Counter()
        k = self.r.rrf_k
        for q in queries:
            for rank, (id_, _) in enumerate(self.store.search_bm25(q, self.r.k_bm25, allowed)):
                rrf[id_] += 1.0 / (k + rank + 1)
            vec = qvec if q is queries[0] else self.embedder.encode([q])[0]
            for rank, (id_, _) in enumerate(self.store.search_dense(vec, self.r.k_dense, allowed)):
                rrf[id_] += 1.0 / (k + rank + 1)
        if not rrf:
            return {}
        # rel(q, h): mostly the dense similarity, with the fused rank as a tie-breaker. The RRF part is
        # rank-based (always 1.0 for the top hit), so it must not dominate or abstention has no signal.
        top = max(rrf.values())
        out = {}
        for id_, s in rrf.items():
            cos = float(np.dot(self.store.vectors.vec(id_), qvec)) if id_ in self.store.vectors else 0.0
            out[id_] = 0.7 * max(cos, 0.0) + 0.3 * (s / top)
        return out

    # ---- validity chains -----------------------------------------------------------------------
    def _resolve_chains(self, fused: dict[str, float], constraint: TemporalConstraint, allowed: set[str],
                        qvec: np.ndarray) -> list[Candidate]:
        out: dict[str, Candidate] = {}
        for id_, rel in fused.items():
            entry = self.store.get(id_)
            if entry is None:
                continue
            if not isinstance(entry, Fact):
                out.setdefault(id_, Candidate(id_, rel))
                continue
            chain = [f for f in self.store.chain(id_) if f.id in allowed]
            if not chain:
                continue
            nodes = self._select_nodes(chain, constraint)
            # With an explicit temporal cue (now / in March / how many times / previously) the
            # selected nodes ARE the answer and replacing the hit is the design (plan section 09).
            # Without one, "trust the tail" has no basis: the hit was retrieved because it matched
            # the question, and dropping it for the tail is what emptied the context on questions
            # like "which did I deal with first" -- so there the hit is kept alongside the tail.
            if constraint.mode == "none" and all(node.id != id_ for node, _ in nodes):
                nodes = [(entry, "hit")] + nodes
            for node, reason in nodes:
                r = rel if node.id == id_ else rel * 0.9
                prev = out.get(node.id)
                if prev is None or prev.rel < r:
                    out[node.id] = Candidate(node.id, r, reason=reason)
        return sorted(out.values(), key=lambda c: -c.rel)

    def _select_nodes(self, chain: list[Fact], constraint: TemporalConstraint) -> list[tuple[Fact, str]]:
        head, tail = chain[0], chain[-1]
        if len(chain) == 1:
            return [(tail, "single")]
        if not self.cfg.write.validity_chain:
            return [(tail, "overwrite")]
        mode = constraint.mode
        if mode == "whole_chain":
            return [(f, "chain") for f in chain]
        if mode == "before_latest":
            return [(chain[-2], "previous"), (tail, "latest")]
        if mode == "interval" and constraint.interval is not None:
            hits = [f for f in chain if constraint.interval.intersects(f.valid_from, f.valid_to)]
            if hits:
                return [(f, "interval") for f in hits]
            earlier = [f for f in chain if f.valid_from <= constraint.interval.end]
            return [(earlier[-1] if earlier else head, "nearest")]
        # "now" and questions without a temporal cue: trust the tail, unless the volatility rule fires
        vol = self.store.volatility(tail.entity, tail.attribute)
        suspect = (vol < _LOW_VOLATILITY and tail.speaker != "user" or tail.kind == "inferred") \
            and tail.value.strip().lower() != head.value.strip().lower()
        if suspect:
            return [(head, "suspect_head"), (tail, "suspect_tail")]
        return [(tail, "latest")]

    # ---- two-hop -------------------------------------------------------------------------------
    def _second_hop(self, cands: list[Candidate], allowed: set[str], qvec: np.ndarray) -> tuple[list[Candidate], int]:
        top = cands[: min(10, len(cands))]
        counts: Counter[str] = Counter()
        for c in top:
            e = self.store.get(c.id)
            for ent in (e.entities if e else []):
                if ent.lower() != "user":
                    counts[ent.lower()] += 1
        if not counts:
            return cands, 0
        rels = [c.rel for c in top]
        confidence = (rels[0] - rels[1]) / max(rels[0], 1e-6) if len(rels) > 1 else 1.0
        share = float(np.clip(1.0 - confidence, 0.2, 0.5))
        hop2_budget = int(self.cfg.budget.read_tokens * share)
        have = {c.id for c in cands}
        extra: dict[str, Candidate] = {}
        for ent, _ in counts.most_common(self.r.second_hop_entities):
            for id_ in self.store.by_entity(ent) & allowed:
                if id_ in have or id_ in extra:
                    continue
                vec = self.store.vectors.vec(id_) if id_ in self.store.vectors else None
                cos = float(np.dot(vec, qvec)) if vec is not None else 0.0
                extra[id_] = Candidate(id_, 0.5 * max(cos, 0.0) + 0.1, hop=2, reason=f"entity:{ent}")
        return cands + sorted(extra.values(), key=lambda c: -c.rel), hop2_budget

    # ---- packing -------------------------------------------------------------------------------
    def _pack(self, cands: list[Candidate], qvec: np.ndarray, hop2_budget: int) -> tuple[list[str], dict[str, float]]:
        R = self.cfg.budget.read_tokens
        rel = {c.id: c.rel for c in cands}
        hop1 = [c.id for c in cands if c.hop == 1]
        hop2 = [c.id for c in cands if c.hop == 2]
        chosen = self._pack_subset(hop1, rel, R - hop2_budget if hop2 else R)
        used = sum(max(self.store.get(i).tokens, 1) for i in chosen)
        if hop2:
            chosen += self._pack_subset(hop2, rel, R - used)
        return chosen, {i: rel[i] for i in chosen}

    def _pack_subset(self, ids: list[str], rel: dict[str, float], budget: int) -> list[str]:
        if not ids or budget <= 0:
            return []
        cost = {i: max(self.store.get(i).tokens, 1) for i in ids}
        policy = self.r.packing
        if policy == "topk":
            out, used = [], 0
            for i in sorted(ids, key=lambda i: -rel[i]):
                if used + cost[i] <= budget:
                    out.append(i)
                    used += cost[i]
            return out
        vecs = {i: self.store.vectors.vec(i) for i in ids if i in self.store.vectors}
        if policy == "mmr":
            lam = self.r.mmr_lambda
            out, used = [], 0
            remaining = [i for i in ids if i in vecs]
            while remaining:
                best, best_score = None, -1e9
                for i in remaining:
                    if used + cost[i] > budget:
                        continue
                    red = max((float(np.dot(vecs[i], vecs[j])) for j in out), default=0.0)
                    score = lam * rel[i] - (1 - lam) * red
                    if score > best_score:
                        best, best_score = i, score
                if best is None or best_score <= 0 and out:
                    break
                out.append(best)
                used += cost[best]
                remaining.remove(best)
            return out
        # budgeted greedy on the read-scale coverage objective
        cov = CoverageState(self.embedder.dim)
        for i in ids:
            if i in vecs:
                cov.add_target(i, vecs[i], rel[i])
        state: list[str] = []

        def gain_fn(item: str, selected: list[str]) -> float:
            # the greedy only ever extends `selected`; a shorter list means "start over"
            if selected[: len(state)] != state:
                for s_id in state:
                    cov.remove(s_id)
                state.clear()
            while len(state) < len(selected):
                nxt = selected[len(state)]
                cov.add(nxt, vecs[nxt])
                state.append(nxt)
            return cov.gain(vecs[item])

        return budgeted_greedy([i for i in ids if i in vecs], cost, gain_fn, budget)

    # ---- abstention ----------------------------------------------------------------------------
    def _should_abstain(self, question: str, packed: list[Entry], top_rel: float) -> bool:
        if not packed:
            return True
        if top_rel >= self.r.tau_abs:
            return False
        q_terms = {t for t in tokenize(question) if t not in _STOP_Q and len(t) > 3}
        packed_terms = set()
        for e in packed:
            packed_terms |= set(tokenize(e.text))
            packed_terms |= {x.lower() for x in e.entities}
        return not (q_terms & packed_terms)
