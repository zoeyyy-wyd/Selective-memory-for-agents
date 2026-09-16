"""Run configuration. Every switch in the plan's ablation table is a field here so that one config
hash identifies a run end-to-end (extraction cache, retrieval, injected content, answers, judge)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Literal

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
    # Cost floor used ONLY when estimating the sieve's scale (max singleton gain per token). Without it a
    # 2-token fact sets the scale at ~56 and the lowest threshold demands gain/token >= 0.19, which
    # ordinary 20-40 token episodes never reach: the store filled to 32% of B and eviction never ran.
    # A candidate's own admission ratio still uses its true cost.
    sieve_cost_floor: int = 20
    hysteresis_gamma: float = 0.1      # γ: swap only if new gain ≥ (1+γ) × victim gain
    episode_dup_sim: float = 0.95      # near-duplicate episodes are merged, not re-stored
    # Every verbatim turn becomes a second-tier entry (Episode.raw) charged to the same budget B: extracted
    # entries are selected exactly as without raw turns; raw turns take whatever budget is left, ranked by
    # specificity x speaker / tokens, are evicted first when entries need room and never displace one.
    # The read path shows surviving raw turns to the reader under budget.raw_tokens.
    raw_turns: bool = False
    validity_chain: bool = True        # False = overwrite (the no_validity_chain ablation)
    # A validity chain means "same key, new value => the old value was superseded". That is true of a
    # single-valued attribute (one location, one employer, one running count) and false of a
    # multi-valued one: a user has many goals, several health issues, dozens of preferences at once.
    # Chaining those turned 16 distinct health facts into one chain whose read-time resolution kept
    # only the tail, so retrieved evidence vanished before packing. Same key + new value on one of
    # these attributes is a new independent fact; exact-value duplicates still merge.
    multi_valued_attributes: list[str] = ["preference", "favorite", "dislike", "allergy", "health", "goal",
                                          "relationship", "method", "other"]


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
    nli_threshold: float = 0.8         # accept a summary if ≥ this fraction of its facts are supported
    # Direction of the entailment gate. "summary_supported": every summary fact must be entailed by
    # some member (member = premise) -- the hallucination check; unsupported facts are dropped.
    # "members_entailed" is the original rule (summary = premise must entail each member); a lossy
    # 1-3-fact summary can never imply the detail in its members, so it rejected 229 of 230
    # summaries on dev and made consolidation a no-op that cost 97% of ingest time.
    nli_direction: Literal["summary_supported", "members_entailed"] = "summary_supported"
    fixed_interval: int = 5


class ReadConfig(BaseModel):
    retrieval: RetrievalName = "two_hop"
    packing: PackingName = "budgeted_greedy"
    k_bm25: int = 30
    turn_weight: float = 1.0   # weight of directly retrieved turns against entry-derived turns in the source expansion
    k_turns: int = 0         # surviving raw turns retrieved directly per query and channel, merged into the source expansion; 0 = only turns behind packed entries
    raw_diversity: int = 0   # source expansion: this many sessions get their best turn before any session gets a second; 0 = pure score order
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
    prompt_version: str = "v3-detail"  # "v3" (original) | "v3-detail" (values keep the stated detail: dev reference 0.52 -> 0.57) | "v3-detail1" (+assistant facts, too costly); part of the cache key
    # Off by default: unlike maxItems it alters sampling for every session, so switching it on
    # invalidates the whole extraction cache. Only reach for it if loops survive maxItems.
    repetition_penalty: float = 1.0
    # Sampled retry when greedy output is degenerate (see LLMExtractor.extract). 0 disables.
    loop_retry_temperature: float = 0.6
    cache_dir: str = ".cache/extract"


class ModelConfig(BaseModel):
    embedder: str = "hash"              # "hash" (offline) or a sentence-transformers name, e.g. BAAI/bge-m3
    embed_dim: int = 256                # used by the hash embedder only
    embed_device: str | None = None     # None = sentence-transformers default (cuda if free); "cpu" if vLLM owns the card
    nli_device: str | None = None       # same, for the NLI cross-encoder
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
    answer_prompt_version: str = "strict"   # "strict" (abstain unless the entries contain the answer) | "grounded" (abstain only if nothing relevant)
    request_max_retries: int = 2      # raise behind a flaky gateway
    request_timeout: float = 600.0
    # Separate, longer read timeout for a self-hosted server (localhost base_url): it queues behind
    # other local jobs rather than hanging like a gateway did, and a 400-token summary can wait
    # minutes when extraction saturates the card.
    local_request_timeout: float = 900.0
    retry_attempts: int = 5           # on top of the SDK's own; total wait grows to ~2 min
    retry_base_delay: float = 2.0
    retry_max_delay: float = 60.0
    nli_model: str = "lexical"          # "lexical" (offline) or a HF cross-encoder, e.g. cross-encoder/nli-deberta-v3-base
    llm_cache_dir: str = ".cache/llm"
    embed_cache_dir: str = ".cache/embed"
    # Per-question ingested state (store + write/consolidation bookkeeping) keyed by the write-side
    # config, so runs that only change the read path, the answerer or the judge skip ingest
    # entirely. None disables. Infra: not part of the run identity.
    ingest_cache_dir: str | None = ".cache/ingest"
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

    # Fields that change how a run is executed but not what it computes: devices, cache locations,
    # timeouts, retries, which env var holds a key. They stay out of the run identity so that editing
    # a timeout mid-campaign, or moving embeddings to the CPU, does not orphan a half-finished run's
    # records (it did: a resumed run landed in a fresh directory and started over).
    INFRA_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "models.embed_device", "models.nli_device", "models.llm_cache_dir", "models.embed_cache_dir",
        "models.ingest_cache_dir", "models.local_request_timeout",
        "models.request_max_retries", "models.request_timeout", "models.retry_attempts",
        "models.retry_base_delay", "models.retry_max_delay", "models.answer_api_key_env", "models.judge_api_key_env",
        "extract.cache_dir", "anthropic.max_retries", "anthropic.timeout",
    })

    @classmethod
    def is_infra_key(cls, dotted: str) -> bool:
        return dotted in cls.INFRA_FIELDS

    def identity(self) -> dict[str, Any]:
        """model_dump with the infra fields removed: everything that can change a result."""
        data = self.model_dump(mode="json")
        for key in self.INFRA_FIELDS:
            node = data
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.get(p, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict):
                node.pop(parts[-1], None)
        return data

    # Everything downstream of ingest. Two runs that differ only here share ingested state.
    READ_SIDE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "read", "anthropic", "budget.read_tokens", "budget.raw_tokens", "models.answer_model", "models.answer_base_url",
        "models.answer_provider", "models.answer_max_tokens", "models.judge_model", "models.judge_base_url",
        "models.judge_provider", "models.judge_max_tokens", "models.answer_prompt_version", "name",
    })

    def ingest_identity(self) -> dict[str, Any]:
        data = self.identity()
        for key in self.READ_SIDE_FIELDS:
            node = data; parts = key.split(".")
            for part in parts[:-1]:
                node = node.get(part, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict):
                node.pop(parts[-1], None)
        return data

    def ingest_key(self, question_id: str) -> str:
        payload = json.dumps({"q": question_id, "cfg": self.ingest_identity()}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def config_hash(self) -> str:
        payload = json.dumps(self.identity(), sort_keys=True)
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
            cfg = cls.model_validate(_load_yaml_tree(Path(path)))
        if overrides:
            cfg = cfg.with_overrides(overrides)
        return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml_tree(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """A config file may name a parent with `extends: other.yaml` (relative to its own directory); the
    child's keys override the parent's, section by section. Without it a partial file silently falls
    back to the code defaults for every field it does not mention."""
    if path in seen:
        raise ValueError(f"config extends cycle: {' -> '.join(str(p) for p in seen + (path,))}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _deep_merge(_load_yaml_tree((path.parent / parent).resolve(), seen + (path,)), data)


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
