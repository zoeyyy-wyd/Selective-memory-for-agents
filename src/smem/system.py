"""Orchestration: the write path (extract -> write policy -> consolidate -> evict) run per session in
time order, and the read path (retrieve -> pack -> answer) per question. Also keeps the bookkeeping
the evidence metrics need: where every candidate came from and what happened to it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from smem.answer import ABSTAIN_TEXT, Answerer, ExtractiveAnswerer, LLMAnswerer, check_citations
from smem.config import SystemConfig
from smem.consolidate import (
    NLI,
    Consolidator,
    HeuristicSummarizer,
    LLMSummarizer,
    Summarizer,
    get_nli,
)
from smem.embed import Embedder, get_embedder
from smem.evict import Evictor
from smem.extract import Extractor, HeuristicExtractor, LLMExtractor
from smem.hawkes import HawkesIntensity
from smem.llm import LLM, OpenAICompatLLM
from smem.read import Reader, ReadResult
from smem.schemas import Fact, Session
from smem.store import MemoryStore, key_drift
from smem.write import WritePolicy


@dataclass
class IngestReport:
    session_id: str
    n_episodes: int
    n_facts: int
    admitted: list[str]
    schema_ok: bool
    store_tokens: int
    store_entries: int


@dataclass
class AskResult:
    question: str
    answer: str
    abstained: bool
    read: ReadResult
    injected_ids: set[str]
    cited_ids: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)


class SelectiveMemory:
    def __init__(
        self,
        cfg: SystemConfig,
        embedder: Embedder | None = None,
        extractor: Extractor | None = None,
        answerer: Answerer | None = None,
        summarizer: Summarizer | None = None,
        nli: NLI | None = None,
        rewriter: LLM | None = None,
        store_path: str = ":memory:",
    ):
        self.cfg = cfg
        self.embedder = embedder or get_embedder(cfg.models.embedder, cfg.models.embed_dim, cfg.models.embed_cache_dir)
        self.extractor = extractor or HeuristicExtractor(cfg.extract.max_episode_tokens)
        self.answerer = answerer or ExtractiveAnswerer()
        self.store = MemoryStore(self.embedder.dim, store_path)
        self.hawkes = HawkesIntensity(cfg.evict.hawkes_alpha, cfg.evict.hawkes_beta, cfg.evict.hawkes_mu_floor)
        self.evictor = Evictor(cfg, self.store, self.hawkes)
        self.writer = WritePolicy(cfg, self.store, self.evictor, self.hawkes, self.embedder.dim)
        self.consolidator = Consolidator(cfg, self.store, self.writer, self.embedder,
                                         summarizer or HeuristicSummarizer(), nli or get_nli(cfg.models.nli_model))
        self.reader = Reader(cfg, self.store, self.embedder, self.hawkes, rewriter)
        self.session_ids: list[str] = []
        self.candidate_origin: dict[str, tuple[str, int]] = {}
        self.alias: dict[str, str] = {}          # candidate id -> id of the stored entry it was merged into
        self.schema_errors = 0
        self.extractions = 0
        self.now: datetime | None = None

    # ---- write path ----------------------------------------------------------------------------
    def ingest_session(self, session: Session) -> IngestReport:
        result = self.extractor.extract(session)
        self.extractions += 1
        if not result.schema_ok:
            self.schema_errors += 1
        self.now = session.ts
        self.session_ids.append(session.session_id)
        for ep in result.episodes:
            self.candidate_origin.setdefault(ep.id, (session.session_id, ep.turn_idx))
        for f in result.facts:
            self.candidate_origin.setdefault(f.id, (session.session_id, -1))
        texts = [e.text for e in result.episodes + result.facts]
        vecs = self.embedder.encode(texts) if texts else np.zeros((0, self.embedder.dim), dtype=np.float32)
        vec_of = {e.id: vecs[i] for i, e in enumerate(result.episodes + result.facts)}
        admitted = self.writer.ingest(result, vec_of, session.ts)
        # merged facts: remember which stored fact now carries this candidate's content
        for f in result.facts:
            if f.id not in self.writer.candidates:
                match = [x for x in self.store.facts_for(f.entity, f.attribute)
                         if x.value.strip().lower() == f.value.strip().lower()]
                if match:
                    self.alias[f.id] = match[0].id
        self.consolidator.after_session(len(self.session_ids) - 1, self.session_ids, session.ts)
        active = self.writer.active()
        return IngestReport(session.session_id, len(result.episodes), len(result.facts), admitted, result.schema_ok,
                            active.used, len(active.selected))

    def ingest(self, sessions: list[Session]) -> list[IngestReport]:
        return [self.ingest_session(s) for s in sorted(sessions, key=lambda s: s.ts)]

    # ---- read path -----------------------------------------------------------------------------
    def ask(self, question: str, now: datetime | None = None) -> AskResult:
        now = now or self.now or datetime.now()
        allowed = self.writer.active_ids()
        read = self.reader.read(question, now, allowed)
        answer = self.answerer.answer(question, read, now)
        abstained = read.abstain or answer.strip().lower().startswith("i don't know")
        cited, invalid = check_citations(answer, read.packed_ids)
        return AskResult(question, answer if not abstained else ABSTAIN_TEXT, abstained, read, read.packed_ids,
                         cited, invalid)

    # ---- bookkeeping for the evidence metrics --------------------------------------------------
    def candidates_from_sessions(self, session_ids: set[str], turn_filter: dict[str, set[int]] | None = None) -> set[str]:
        """Ids of every candidate extracted from the given sessions (resolving merges to the stored id).
        turn_filter restricts episodes to turns with has_answer; facts are kept regardless."""
        out = set()
        for cid, (sid, turn) in self.candidate_origin.items():
            if sid not in session_ids:
                continue
            if turn_filter is not None and sid in turn_filter and turn >= 0 and turn not in turn_filter[sid]:
                continue
            out.add(self.alias.get(cid, cid))
        return out

    def surviving_ids(self) -> set[str]:
        return self.writer.active_ids()

    def summary_sources(self) -> dict[str, set[str]]:
        return self.consolidator.summary_sources

    def stats(self) -> dict:
        active = self.writer.active()
        return {
            "store_tokens": active.used,
            "store_entries": len(active.selected),
            "pool_entries": len(self.store),
            "n_sieves": len(self.writer.sieves),
            "active_threshold": active.threshold,
            "write": self.writer.stats.as_dict(),
            "consolidation": self.consolidator.stats.as_dict(),
            "extractions": self.extractions,
            "schema_errors": self.schema_errors,
            "key_drift": self.key_drift_summary(),
        }

    def key_drift_summary(self, examples: int = 10) -> dict:
        drift = key_drift(e for e in self.writer.candidates.values() if isinstance(e, Fact))
        return {
            "n_entities": drift["n_entities"],
            "n_keys": drift["n_keys"],
            "n_suspicious_attribute_pairs": len(drift["suspicious_attribute_pairs"]),
            "n_suspicious_entity_pairs": len(drift["suspicious_entity_pairs"]),
            "examples": (drift["suspicious_attribute_pairs"] + drift["suspicious_entity_pairs"])[:examples],
        }

    def close(self) -> None:
        self.store.close()


def build_llm(model: str, base_url: str | None, cfg: SystemConfig, constrained: bool = True) -> OpenAICompatLLM:
    return OpenAICompatLLM(model, base_url=base_url, cache_dir=cfg.models.llm_cache_dir, constrained_decoding=constrained)


def build_system(cfg: SystemConfig, backend: str = "offline", store_path: str = ":memory:") -> SelectiveMemory:
    """backend='offline' uses the heuristic extractor / summariser / answerer; backend='llm' wires the
    models named in cfg.models (a vLLM server for extraction, an OpenAI-compatible answering model)."""
    if backend == "offline":
        return SelectiveMemory(cfg, store_path=store_path)
    constrained = cfg.extract.constrained_decoding
    extract_llm = build_llm(cfg.models.extract_model, cfg.models.extract_base_url, cfg, constrained)
    answer_llm = build_llm(cfg.models.answer_model, cfg.models.answer_base_url, cfg)
    return SelectiveMemory(
        cfg,
        extractor=LLMExtractor(extract_llm, cfg.extract.cache_dir, constrained, cfg.extract.max_episode_tokens),
        answerer=LLMAnswerer(answer_llm),
        summarizer=LLMSummarizer(extract_llm, constrained),
        rewriter=extract_llm,
        store_path=store_path,
    )
