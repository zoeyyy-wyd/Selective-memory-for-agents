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
ProviderName = Literal["auto", "openai", "anthropic"]


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
    # 2048 truncated 190 of 4564 dev sessions mid-JSON (4.2%): rich sessions produce more
    # episodes+facts than that. Not part of the extraction cache key on purpose -- at
    # temperature 0 an output that finished on its own is byte-identical under a larger cap,
    # so only the truncated entries are stale and the rest stay valid.
    max_output_tokens: int = 4096
    # Off by default: unlike maxItems it alters sampling for every session, so switching it on
    # invalidates the whole extraction cache. Only reach for it if loops survive maxItems.
    repetition_penalty: float = 1.0
    # Sampled retry when greedy output is degenerate (see LLMExtractor.extract). 0 disables.
    loop_retry_temperature: float = 0.6
    cache_dir: str = ".cache/extract"


class ModelConfig(BaseModel):
    embedder: str = "hash"              # "hash" (offline) or a sentence-transformers name, e.g. BAAI/bge-m3
    embed_dim: int = 256                # used by the hash embedder only
    extract_model: str = "Qwen/Qwen3-8B-AWQ"
    extract_base_url: str = "http://localhost:8000/v1"
    answer_model: str = "gpt-4.1-mini"
    answer_base_url: str | None = None
    judge_model: str = "gpt-4.1-mini"
    judge_base_url: str | None = None
    # The judge answers in a word, but a reasoning judge spends its budget before any text; 10 was
    # enough for gpt-4.1-mini and returns an empty string on gpt-5-mini. See LLMJudge.
    judge_max_tokens: int = 512
    # The answer is one or two sentences, but a reasoning answerer spends its budget thinking first:
    # glm-5.3 used ~500 tokens on a temporal question and would return "" at the old 300.
    answer_max_tokens: int = 300
    request_max_retries: int = 2      # raise behind a flaky gateway
    request_timeout: float = 600.0
    retry_attempts: int = 5           # on top of the SDK's own; total wait grows to ~2 min
    retry_base_delay: float = 2.0
    retry_max_delay: float = 60.0
    nli_model: str = "lexical"          # "lexical" (offline) or a HF cross-encoder, e.g. cross-encoder/nli-deberta-v3-base
    llm_cache_dir: str = ".cache/llm"
    embed_cache_dir: str = ".cache/embed"
    schema_mode: Literal["response_format", "guided_json"] = "response_format"
    # "auto" sends claude-* models to the Anthropic SDK and everything else to the OpenAI-compatible
    # client. Extraction is deliberately not routed here: it stays on the local vLLM server.
    answer_provider: ProviderName = "auto"
    judge_provider: ProviderName = "auto"
    # Name of the env var holding each role's key. Two OpenAI-format clients can need two keys:
    # answering through a gateway (TOKENROUTER_API_KEY) while the judge goes to api.openai.com
    # directly (OPENAI_API_KEY) because the official LongMemEval judge, GPT-4o, is not on the gateway.
    answer_api_key_env: str = "OPENAI_API_KEY"
    judge_api_key_env: str = "OPENAI_API_KEY"
    # sent only to self-hosted servers (vLLM / llama.cpp); e.g. switch off Qwen3 thinking
    extract_extra_body: dict[str, Any] = Field(default_factory=dict)


class AnthropicConfig(BaseModel):
    """Applies to whichever calls route to the Anthropic SDK. Both answering and judging want the
    model to read the injected memory, not to reason its way around a gap in it: thinking off (or
    low effort) keeps the measurement about the memory system rather than the reader."""

    thinking: Literal["off", "adaptive"] = "off"
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # Re-runs a declined request on another model inside the same call. Off by default: in an
    # ablation it would silently mix two answering models into one run. See AnthropicLLM.
    fallback_model: str | None = None
    max_retries: int = 2
    timeout: float = 600.0


class SystemConfig(BaseModel):
    name: str = "default"
    budget: Budget = Field(default_factory=Budget)
    write: WriteConfig = Field(default_factory=WriteConfig)
    evict: EvictConfig = Field(default_factory=EvictConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    read: ReadConfig = Field(default_factory=ReadConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)

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


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Read KEY=value lines from a .env file into os.environ without overwriting anything already
    exported. Stdlib only; keys live in a gitignored file rather than on the command line."""
    import os

    p = Path(path)
    if not p.exists():
        return []
    loaded = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
