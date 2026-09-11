"""Run configuration. Every switch in the plan's ablation table is a field here so that one config
hash identifies a run end-to-end (extraction cache, retrieval, injected content, answers, judge)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from smem.schemas import Budget

WritePolicyName = Literal["all", "novelty_threshold", "sieve"]
EvictPolicyName = Literal["fifo", "lru", "random", "utility_heuristic", "swap", "swap_no_hawkes"]
ConsolidationName = Literal["redundancy", "fixed_interval", "no_consolidation", "no_nli_check"]
RetrievalName = Literal["single_hop", "two_hop"]
PackingName = Literal["topk", "mmr", "budgeted_greedy"]


class WriteConfig(BaseModel):
    policy: WritePolicyName = "sieve"
    novelty_threshold: float = 0.15    # v1 heuristic: skip if 1 - max_sim < this
    sieve_eps: float = 1.0             # geometric threshold grid (1+eps)^i for SieveStreaming
    sieve_min_threshold: float = 0.0   # lowest threshold as a fraction of the max gain/token; 0 = auto (≈ cost/2B)
    sieve_max_threshold: float = 0.5
    hysteresis_gamma: float = 0.1      # γ: swap only if new gain ≥ (1+γ) × victim gain
    episode_dup_sim: float = 0.95      # near-duplicate episodes are merged, not re-stored
    validity_chain: bool = True        # False = overwrite (the no_validity_chain ablation)


class EvictConfig(BaseModel):
    policy: EvictPolicyName = "swap"
    hawkes_mu_floor: float = 0.01      # base intensity floor so weights never reach zero
    hawkes_alpha: float = 0.5
    hawkes_beta: float = 0.03          # per day (half-life ≈ 23 days); refit on dev
    heuristic_weights: dict[str, float] = Field(
        default_factory=lambda: {"recency": 0.4, "frequency": 0.3, "specificity": 0.2, "source": 0.1}
    )
    seed: int = 0


class ConsolidationConfig(BaseModel):
    policy: ConsolidationName = "redundancy"
    cluster_sim: float = 0.6           # threshold clustering on cosine similarity
    window_sessions: int = 10          # only cluster episodes from the last K sessions
    min_cluster_size: int = 3
    redundancy_threshold: float = 0.35 # fires when mean Δf(h|S∖h) per member falls below this
    nli_threshold: float = 0.8         # accept a summary if ≥ this fraction of members are entailed
    fixed_interval: int = 5


class ReadConfig(BaseModel):
    retrieval: RetrievalName = "two_hop"
    packing: PackingName = "budgeted_greedy"
    k_bm25: int = 30
    k_dense: int = 30
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    second_hop_entities: int = 5
    tau_abs: float = 0.3              # abstention threshold on the top packed relevance (tuned on dev)
    rewrite_queries: bool = True


class ExtractConfig(BaseModel):
    backend: Literal["heuristic", "llm"] = "heuristic"
    constrained_decoding: bool = True
    max_episode_tokens: int = 60
    max_turn_tokens: int = 2000        # long ShareGPT essays/code are cut per turn before extraction
    cache_dir: str = ".cache/extract"


class ModelConfig(BaseModel):
    embedder: str = "hash"              # "hash" (offline) or a sentence-transformers name, e.g. BAAI/bge-m3
    embed_dim: int = 256                # used by the hash embedder only
    extract_model: str = "Qwen/Qwen3-8B-AWQ"
    extract_base_url: str = "http://localhost:8000/v1"
    answer_model: str = "gpt-4.1-mini"
    answer_base_url: str | None = None
    judge_model: str = "gpt-4.1-mini"
    nli_model: str = "lexical"          # "lexical" (offline) or a HF cross-encoder, e.g. cross-encoder/nli-deberta-v3-base
    llm_cache_dir: str = ".cache/llm"
    embed_cache_dir: str = ".cache/embed"
    schema_mode: Literal["response_format", "guided_json"] = "response_format"
    # sent only to self-hosted servers (vLLM / llama.cpp); e.g. switch off Qwen3 thinking
    extract_extra_body: dict[str, Any] = Field(default_factory=dict)


class SystemConfig(BaseModel):
    name: str = "default"
    budget: Budget = Field(default_factory=Budget)
    write: WriteConfig = Field(default_factory=WriteConfig)
    evict: EvictConfig = Field(default_factory=EvictConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    read: ReadConfig = Field(default_factory=ReadConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def with_overrides(self, overrides: dict[str, Any]) -> SystemConfig:
        """Apply dotted overrides such as {"write.policy": "all", "budget.read_tokens": 4000}."""
        data = self.model_dump()
        for key, value in overrides.items():
            node = data
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return SystemConfig.model_validate(data)

    @classmethod
    def load(cls, path: str | Path | None, overrides: dict[str, Any] | None = None) -> SystemConfig:
        cfg = cls()
        if path is not None:
            with open(path) as f:
                cfg = cls.model_validate(yaml.safe_load(f) or {})
        if overrides:
            cfg = cfg.with_overrides(overrides)
        return cfg


def parse_override(text: str) -> tuple[str, Any]:
    """'write.policy=sieve' -> ('write.policy', 'sieve'); values are YAML-parsed so numbers/bools work."""
    key, _, raw = text.partition("=")
    if not _:
        raise ValueError(f"override must look like key=value, got {text!r}")
    return key.strip(), yaml.safe_load(raw.strip())
