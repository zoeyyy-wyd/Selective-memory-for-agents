"""Consolidation (plan section 08): redundancy-triggered, entailment-verified.

After each session the new unconsolidated episodes are threshold-clustered; a cluster fires when its
members' mean normalised coverage loss Δf(h | S∖h) / w_h is small (they cover each other). The
summariser compresses the cluster into 1–3 facts; a summary is accepted only if at least
`nli_threshold` of the members are entailed by it. Accepted summaries enter the store through the
write policy (so the budget still applies); members are tagged consolidated and go first at swap time."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from smem.config import SystemConfig
from smem.embed import Embedder, tokenize
from smem.llm import LLM, parse_json_object
from smem.schemas import Episode, Fact, make_id
from smem.store import MemoryStore
from smem.tokens import count_tokens
from smem.write import WritePolicy

_STOP = frozenset({"a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them", "this", "that", "these", "those", "there", "here", "have", "has", "had", "do", "does", "did", "not", "no", "yes", "just", "very", "really", "also", "about", "into", "over", "after", "before", "while", "when", "where", "what", "which", "who", "whom", "how", "why", "can", "could", "would", "should", "will", "shall", "may", "might", "must"})


def content_words(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 2}


# ---- summarisers -------------------------------------------------------------------------------
class Summarizer(Protocol):
    def summarize(self, members: list[Episode], vecs: np.ndarray) -> list[Fact]: ...


class HeuristicSummarizer:
    """Medoid of the cluster becomes one 'recurring_topic' fact. No model; keeps the offline path
    runnable and is the repair fallback for the LLM summariser."""

    def summarize(self, members: list[Episode], vecs: np.ndarray) -> list[Fact]:
        if not members:
            return []
        sims = vecs @ vecs.T
        medoid = int(np.argmax(sims.sum(axis=1)))
        m = members[medoid]
        ents = Counter(e for ep in members for e in ep.entities if e.lower() != "user")
        entity = "user" if any(ep.speaker == "user" for ep in members) else "assistant"
        topic = ents.most_common(1)[0][0].lower().replace(" ", "_") if ents else "general"
        value = m.text
        return [_summary_fact(entity, f"recurring_{topic}", value, members)]


SUMMARY_PROMPT = """You compress a cluster of related memory episodes about a user into 1 to 3 durable facts.
Each fact is (entity, attribute, value). Keep every concrete detail that a later question could ask about
(names, places, dates, numbers). Do not add anything the episodes do not say. Output JSON:
{"facts": [{"entity": "user", "attribute": "snake_case", "value": "..."}]}"""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "object", "properties": {
        "entity": {"type": "string"}, "attribute": {"type": "string"}, "value": {"type": "string"}},
        "required": ["entity", "attribute", "value"]}, "maxItems": 3}},
    "required": ["facts"],
}


class LLMSummarizer:
    def __init__(self, llm: LLM, constrained_decoding: bool = True):
        self.llm = llm
        self.constrained_decoding = constrained_decoding
        self.fallback = HeuristicSummarizer()
        self.n_schema_errors = 0

    def summarize(self, members: list[Episode], vecs: np.ndarray) -> list[Fact]:
        body = "\n".join(f"- ({ep.ts.date()}, {ep.speaker}) {ep.text}" for ep in members)
        raw = self.llm.complete(SUMMARY_PROMPT, body, json_schema=SUMMARY_SCHEMA if self.constrained_decoding else None,
                                max_tokens=400)
        obj = parse_json_object(raw)
        if not obj or not isinstance(obj.get("facts"), list):
            self.n_schema_errors += 1
            return self.fallback.summarize(members, vecs)
        out = []
        for item in obj["facts"][:3]:
            try:
                attr = re.sub(r"[^a-z0-9_]+", "_", str(item["attribute"]).lower()).strip("_") or "summary"
                out.append(_summary_fact(str(item["entity"]) or "user", attr, str(item["value"]), members))
            except (KeyError, TypeError):
                self.n_schema_errors += 1
        return out or self.fallback.summarize(members, vecs)


def _summary_fact(entity: str, attribute: str, value: str, members: list[Episode]) -> Fact:
    latest = max(members, key=lambda e: e.ts)
    text = f"{entity} {attribute}: {value}"
    return Fact(
        id=make_id("fs", entity, attribute, value, latest.session_id), entity=entity, attribute=attribute, value=value,
        valid_from=min(e.ts for e in members), sources=[e.id for e in members], kind="inferred",
        speaker="user" if any(e.speaker == "user" for e in members) else "assistant",
        session_id=latest.session_id, tokens=count_tokens(text),
    )


# ---- entailment --------------------------------------------------------------------------------
class NLI(Protocol):
    def entails(self, premise: str, hypothesis: str) -> float: ...


class LexicalNLI:
    """Fraction of the hypothesis' content words present in the premise. A stand-in for the
    DeBERTa cross-encoder so the pipeline (and the accept/reject bookkeeping) runs offline."""

    def entails(self, premise: str, hypothesis: str) -> float:
        hw = content_words(hypothesis)
        if not hw:
            return 1.0
        pw = content_words(premise)
        return len(hw & pw) / len(hw)


class CrossEncoderNLI:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str | None = None):
        from transformers import pipeline  # heavy import kept local

        # device=None lets transformers pick (cuda if present); "cpu" when vLLM owns the card
        self.pipe = pipeline("text-classification", model=model_name, top_k=None,
                             **({"device": device} if device else {}))

    def entails(self, premise: str, hypothesis: str) -> float:
        scores = self.pipe({"text": premise, "text_pair": hypothesis})
        for s in scores:
            if s["label"].lower().startswith("entail"):
                return float(s["score"])
        return 0.0


def get_nli(name: str, device: str | None = None) -> NLI:
    return LexicalNLI() if name == "lexical" else CrossEncoderNLI(name, device=device)


# ---- consolidator ------------------------------------------------------------------------------
@dataclass
class ConsolidationStats:
    runs: int = 0
    clusters_checked: int = 0
    clusters_fired: int = 0
    summaries_accepted: int = 0
    summaries_rejected: int = 0
    episodes_consolidated: int = 0
    summary_facts_admitted: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


class Consolidator:
    def __init__(self, cfg: SystemConfig, store: MemoryStore, writer: WritePolicy, embedder: Embedder,
                 summarizer: Summarizer, nli: NLI):
        self.cfg = cfg
        self.c = cfg.consolidation
        self.store = store
        self.writer = writer
        self.embedder = embedder
        self.summarizer = summarizer
        self.nli = nli
        self.stats = ConsolidationStats()
        self.summary_sources: dict[str, set[str]] = {}   # summary fact id -> member episode ids
        self.log: list[dict] = []

    def after_session(self, session_idx: int, recent_session_ids: list[str], now: datetime) -> None:
        if self.c.policy == "no_consolidation":
            return
        if self.c.policy == "fixed_interval" and (session_idx + 1) % self.c.fixed_interval != 0:
            return
        self.stats.runs += 1
        window = set(recent_session_ids[-self.c.window_sessions:])
        sieve = self.writer.active()
        members = [self.store.episodes[i] for i in sieve.selected
                   if i in self.store.episodes and not self.store.episodes[i].consolidated
                   and self.store.episodes[i].session_id in window]
        if len(members) < self.c.min_cluster_size:
            return
        vecs = np.stack([np.asarray(e.embedding, dtype=np.float32) for e in members])
        for cluster in threshold_clusters(vecs, self.c.cluster_sim):
            if len(cluster) < self.c.min_cluster_size:
                continue
            self.stats.clusters_checked += 1
            eps = [members[i] for i in cluster]
            if self.c.policy != "fixed_interval" and not self._is_redundant(sieve, eps):
                continue
            self.stats.clusters_fired += 1
            self._compress(eps, vecs[cluster], now)

    def _is_redundant(self, sieve, eps: list[Episode]) -> bool:
        losses = sieve.cov.losses()
        vals = []
        for e in eps:
            w = max(sieve.cov.weight(e.id), 1e-6) if sieve.cov.has_target(e.id) else 1.0
            vals.append(losses.get(e.id, 0.0) / w)
        return float(np.mean(vals)) < self.c.redundancy_threshold

    def _compress(self, eps: list[Episode], vecs: np.ndarray, now: datetime) -> None:
        facts = self.summarizer.summarize(eps, vecs)
        if not facts:
            self.stats.summaries_rejected += 1
            return
        summary_text = " ".join(f.text for f in facts)
        if self.c.policy != "no_nli_check":
            entailed = sum(1 for e in eps if self.nli.entails(summary_text, e.text) >= 0.5)
            rate = entailed / len(eps)
            if rate < self.c.nli_threshold:
                self.stats.summaries_rejected += 1
                self.log.append({"accepted": False, "rate": rate, "members": [e.id for e in eps]})
                return
        else:
            rate = 1.0
        self.stats.summaries_accepted += 1
        fvecs = self.embedder.encode([f.text for f in facts])
        admitted_any = False
        for f, v in zip(facts, fvecs):
            if self.writer.admit_external(f, v, now):
                admitted_any = True
                self.stats.summary_facts_admitted += 1
                self.summary_sources[f.id] = {e.id for e in eps}
        if admitted_any:
            ids = [e.id for e in eps]
            self.store.mark_consolidated(ids)
            self.stats.episodes_consolidated += len(ids)
        self.log.append({"accepted": True, "rate": rate, "members": [e.id for e in eps],
                         "facts": [f.id for f in facts], "admitted": admitted_any})


def threshold_clusters(vecs: np.ndarray, sim_threshold: float) -> list[list[int]]:
    """Single-link clustering: i ~ j if cos(i, j) ≥ threshold. Union-find over the similarity matrix."""
    n = len(vecs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sims = vecs @ vecs.T
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= sim_threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


