# A Selective Memory System for Long-Conversation Agents

An external long-term memory for a long-conversation agent whose capacity is **bounded**. Under a
storage budget *B* (total memory tokens) and a read budget *R* (tokens injected per query) the system
decides what is worth writing, what to forget when space runs out, and how much to read each time.
Write, forget and read are one budgeted submodular coverage problem, with a Hawkes-process access prior
and entailment-verified consolidation. Nothing is trained. The deliverable on LongMemEval is the
accuracy-vs-budget curve plus three metrics that measure writing, forgetting and reading directly.

The original technical plan is [selective-memory-for-long-conversation-agents.md](selective-memory-for-long-conversation-agents.md).
This repository implements it end to end and has been run on LongMemEval-S (September 2026). Where the
implementation departs from the plan, the plan is superseded by what is written here and in `docs/`.

**Headline numbers** (gpt-4.1-mini reader, GPT-4o judge, `configs/raw_turns.yaml`): unbounded store
**0.80 on 297 questions** (dev100 0.76, held-out test200 0.82); accuracy-vs-budget at B = 1.5% / 3% /
12.5% / 25% of the history: 0.26 / 0.38 / 0.60 / 0.76 with swap eviction, vs 0.22 / 0.27 / 0.60 / – with FIFO.

Documentation (Chinese):

| file | what it is |
|---|---|
| [docs/results.md](docs/results.md) | every result: final config on dev / test / combined, budget curves, every variant tried and why it was or was not adopted |
| [docs/process.md](docs/process.md) | the workflow: what the system does step by step, the timeline of the work, how each change was verified, how to reproduce |
| [docs/accuracy-plan.md](docs/accuracy-plan.md) | the accuracy work in detail: ceiling, failure attribution, the read-side experiments, raw turns as budgeted entries, held-out check |
| [docs/memory-systems-survey.md](docs/memory-systems-survey.md) | what the 80%+ LongMemEval systems do, and what of it applies here |
| [docs/speed-optimization.md](docs/speed-optimization.md) | how a 100-question evaluation went from hours to minutes |

## Setup

```bash
conda env create -f environment.yml      # python 3.11, faiss-cpu, torch, sentence-transformers
conda activate smem                      # (the env already has `pip install -e .[dev]` applied)
pytest                                   # 85 tests, all offline, ~5 s
```

Two backends exist for every model-dependent step so that the whole pipeline runs without a GPU or an
API key:

| step | `--backend offline` (tests, demo, dry runs) | `--backend llm` (real experiments) |
|---|---|---|
| extraction | rule-based `HeuristicExtractor` | Qwen3-8B via vLLM, schema-constrained (`guided_json`) |
| embeddings | feature-hashing `HashEmbedder` | `BAAI/bge-m3` via sentence-transformers, cached on disk |
| consolidation summary | medoid episode | same local model, JSON schema |
| entailment check | lexical overlap | `cross-encoder/nli-deberta-v3-base` |
| answering | extractive stand-in | `gpt-4.1-mini`, or Claude via `configs/claude.yaml` |
| judge | exact match | port of the official LongMemEval judge prompts |

## The worked example (plan section 01)

```bash
smem-demo --conversation demo/sample_conversation.json --store-budget 260 --read-budget 120 --now 2023-06-10 \
  --ask "Which city does the user live in now?" --ask "Where did the user live in March?" \
  --ask "How many times did the user move?"
```

The demo prints, per session, what was extracted, what was admitted, and what was swapped out; then
the validity chains (`Boston [2023-02-04..2023-05-19] -> Seattle [2023-05-19..now]`); then, per question,
the temporal constraint, the injected entries with their relevance, and the cited answer. "now" resolves
to the chain tail (Seattle), "in March" to the interval node (Boston), "how many times" to the whole chain.

## Layout

```
src/smem/
  schemas.py      Episode / Fact / Budget / Session (plan §04)
  coverage.py     the one objective: incremental facility-location coverage + KMN budgeted greedy (§05, §09)
  hawkes.py       per-entity Hawkes intensity, MLE grid fit, specificity/source priors (§05)
  extract.py      heuristic + LLM extractors, constrained decoding, date normalisation, session cache (§06)
  write.py        dedup / update, SieveStreaming thresholds, swap admission, chains as units (§06)
  evict.py        fifo / lru / random / utility_heuristic controls and the swap rule (§06)
  store.py        SQLite + FAISS (numpy fallback) + BM25; validity-chain operations; volatility (§04, §07)
  temporal.py     temporal constraint parsing and interval algebra (§07)
  consolidate.py  redundancy-triggered clustering, LLM/medoid summaries, NLI acceptance (§08)
  read.py         rewrite, hybrid retrieval (RRF), chain resolution, two-hop, topk/MMR/greedy packing, abstention (§09)
  answer.py       injection format, answering prompt, id-citation check (§09)
  system.py       SelectiveMemory orchestration + evidence bookkeeping; build_system(cfg, backend)
  embed.py, llm.py, config.py, tokens.py
evals/
  longmemeval/    loader (HF download), stratified dev/test split, judge wrapper
  metrics.py      evidence write / survival / injection rates, abstention, cost, bootstrap CIs (§10)
  run.py          smem-eval CLI; results/<run id>/{config.yaml, records.jsonl, summary.json}
  extract_cache.py  smem-extract: parallel one-time extraction of every unique session
  ablations.yaml  one group per design decision + the two budget sweeps
baselines/        oracle, full_context, naive_rag, mem0_oss (§02)
configs/          default.yaml (real models); raw_turns.yaml (final: default + verbatim turns, `extends:`);
                  offline.yaml (no models); claude.yaml, tokenrouter.yaml (other answering backends)
scripts/          prewarm_embeddings.py (batch-embed entries / raw turns on the GPU before an evaluation)
docs/             results, process, experiment logs (see the table above)
demo/cli.py       smem-demo
tests/            validity chains, coverage vs brute force, KMN packing, eviction order, sieve budgets,
                  temporal parsing, extraction schema errors, consolidation triggers, end-to-end, harness
```

## How the pieces fit

**One objective.** `f_W(S) = Σ_h w_h · max_{s∈S} sim(h, s)`. `CoverageState` keeps, for every target
*h*, the best and second-best similarity over the selected set and which member provides each, so a
marginal gain is one matrix-vector product and all per-member losses come out of one `bincount`.
At the storage scale *H* is every candidate seen so far and *S* the store; at the read scale *H* is the
retrieval candidate set and *S* the packed context.

**Keys.** A fact's key is `(entity, attribute)`. The long tail lives in the entity (a snake_case noun
phrase naming the thing described: `charity_5k_run`, `korean_restaurants_tried`, `rachel`); the attribute
comes from a closed list of ~25 names (`location`, `count`, `personal_best`, `setting`, ...) that
constrained decoding pins, with `other` + `custom_attribute` as the escape hatch. This is what makes two
mentions of the same thing land on the same key so a validity chain can form. 42 of the 72 answerable
knowledge-update questions are running counts, so the prompt asks for the current total, never the
increment. Residual drift (`location` vs `current_location`, `rachel` vs `rachel_colleague`) is
reported per run under `key_drift` in the stats and never auto-merged: a wrong merge closes a valid
fact, a missed merge only leaves two independent facts.

**Write (`write.py`).** Facts are first deduplicated (same value → union sources) or turned into a
knowledge update (same entity/attribute, new value → validity chain; the `no_validity_chain` ablation
overwrites). Truly new candidates then go through streaming selection: `sieve` keeps a geometric grid
of admission thresholds on gain per token, each with its own candidate store; the store with the largest
coverage value is the active one. Physical entries live once, reference-counted across sieves.
When a store is full, the eviction policy decides: `swap` picks the unit with the smallest coverage loss
per token (consolidated episodes first; a chain is one unit) and admits the candidate only if its gain
per token beats the victim's by `(1 + γ)`; `fifo`/`lru`/`random`/`utility_heuristic` make room
unconditionally.

**Raw turns (`write.raw_turns`, the final configuration).** Every verbatim conversation turn is also
stored, as a second-tier entry charged to the same budget *B*: extracted entries are selected exactly as
without raw turns (raw turns are not coverage targets, do not touch the sieve normaliser and do not
count as Hawkes mentions), raw turns fill whatever budget is left, ranked by `spec · src / max(tokens, 20)`,
are evicted first when an entry needs room and never displace one. With *B* unbounded every turn is
kept; at *B* ≤ 6% of the history none survives and the system reduces to the extracted-entries one.
This is what lifted dev100 from 59 to 76-77: the reader sees the source text, the extracted facts act as
its index and carry the time semantics.

**Weights.** `w_h = spec(h) · src(h) · λ_e(t)` with `λ_e` the Hawkes intensity of the entry's hottest
key: facts belong to their `entity/attribute` process (e.g. `user/city`), episodes to their named
entities; mentions and retrieval hits are the events. `swap_no_hawkes` sets `λ = 1`.

**Read (`read.py`).** Temporal constraint → 1–2 queries → BM25 ⊕ dense fused with RRF → every fact hit
is replaced by the chain node(s) its interval intersects ("now" = tail, subject to the volatility rule;
change questions = whole chain) → optional entity two-hop with an adaptive budget share → packing under
*R* with KMN budgeted greedy + CELF (MMR and top-k are the controls) → abstain when the top relevance is
below `τ_abs` *and* no question term appears in the packed content. With raw turns on, the packed entries
are followed by up to `budget.raw_tokens` (6k) of surviving verbatim turns: the turns the packed entries were
extracted from (neighbours at half weight) plus, with `read.k_turns`, turns retrieved directly (dense over the
turn text, BM25 over turn text + the facts extracted from it). The two are shown to the reader as separate
sections; entry ids and raw-turn ids are both citable.

**Consolidation (`consolidate.py`).** After each session, unconsolidated episodes from the last *K*
sessions are threshold-clustered; a cluster fires when its members' mean normalised coverage loss is
small (they cover each other). The summary is accepted only if ≥ `nli_threshold` of the members are
entailed by it; accepted summaries are admitted through the write policy and members become first-out.

**Measurement (`evals/metrics.py`).** For each question, *E* = the candidates extracted from the
annotated evidence sessions (episodes restricted to `has_answer` turns). Write rate = |E ∩ written| / |E|,
survival rate = |E ∩ store| / |E| (lenient variant counts members of an accepted stored summary),
injection rate = |E ∩ injected| / |E|. Together with the judge label they attribute every wrong answer to
one module.

## Running the evaluation

```bash
# dataset statistics (downloads LongMemEval-S from HuggingFace on first use → data/)
smem-eval --stats-only --split all
# 500 questions; 23867 session slots; 18464 unique sessions; mean history tokens 103064

# offline dry run of the whole harness on 20 dev questions
smem-eval --split dev --config configs/offline.yaml --backend offline --judge exact \
          --store-budget 0.25 --read-budget 2000 --limit 20

# real run (needs a vLLM server for extraction and OPENAI_API_KEY for answering/judging)
smem-eval --split dev --config configs/default.yaml --backend llm --judge llm --store-budget 0.25 --read-budget 2000

# one ablation group, one baseline
smem-eval --split dev --ablation-group evict --store-budget 0.25 --backend llm --judge llm
smem-eval --split dev --baseline naive_rag --read-budget 2000 --backend llm --judge llm
```

`--store-budget` ≤ 1 is a fraction of the question's history tokens, > 1 an absolute token count.
Every run is keyed by a hash of (config, budgets, backend, split, baseline, overrides); rerunning
reuses existing records, `--rerun` recomputes. Answers, injected ids, store statistics and judge labels
are all in `records.jsonl`; `summary.json` has every aggregate with a bootstrap 95% CI.

### Answering on Claude

`configs/claude.yaml` moves **only** the answering model to Claude. Extraction, embeddings and the NLI
check stay on the L4: Anthropic has no embedding endpoint, and Qwen3-8B extraction is already free.
The judge stays on the OpenAI models on purpose — `evals/longmemeval/judge.py` is a port of the
official `evaluate_qa.py`, and swapping the judge makes accuracy incomparable with published
LongMemEval numbers. Claude is worth running as a *second* judge for an agreement check.

```bash
pip install -e ".[claude]"          # or: uv pip install -e ".[dev,claude]"
export ANTHROPIC_API_KEY=...        # or put it in .env, or run `ant auth login`

# step 3 of the build order: pick the answering model on 50 dev questions (~$0.60)
for m in gpt-4.1-mini claude-sonnet-5 claude-haiku-4-5; do
  smem-eval --split dev --config configs/claude.yaml --backend llm --judge llm \
            --limit 50 --read-budget 2000 --ablate models.answer_model=$m
done
```

`answer_provider: auto` sends `claude-*` names to the Anthropic SDK and everything else to the
OpenAI-compatible client; a `base_url` always wins, so extraction stays pinned to vLLM. Three Claude
behaviours the backend absorbs so the rest of the pipeline does not have to know about them:
`temperature` is gone on Claude 4.6+ (rerun determinism comes from `DiskCache`, as before), structured
output is `output_config.format`, and thinking is **off by default** — it is on by default on Sonnet 5
and Opus 5, and its tokens come out of `max_tokens`, which would leave the judge's `max_tokens=10`
call with no text at all. Thinking off is also the right experimental choice: a reader that reasons
around a retrieval gap masks exactly the retrieval differences the budget curve is meant to expose.

### Running the real experiments

1. Serve the extraction model on the GPU box:
   `vllm serve Qwen/Qwen3-8B-AWQ --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.75`
   (0.75 leaves room for bge-m3 next to it; `configs/default.yaml` switches Qwen3 thinking off and sends
   the schema as `response_format`).
2. Prefill the extraction cache, 16 workers (32 saturates the KV cache and is slower):
   `smem-extract --split dev --workers 16 --ablate extract.prompt_version=v3-detail`. About 4 s per
   session: dev's 4564 unique sessions take ~6.7 h, the 197-question test subset's 8510 sessions ~9.5 h.
   The cache is keyed by prompt version, so versions coexist.
3. Prewarm the embedding cache on the GPU, otherwise the evaluation computes them on the CPU:
   `python scripts/prewarm_embeddings.py --split dev --prompt-version v3-detail` and, for raw turns,
   `python scripts/prewarm_embeddings.py --split dev --turns --device cuda --batch 8`.
4. `export OPENAI_API_KEY=...` (or `.env`). Evaluate in three shards (six shards hit the 200k tokens/min
   limit), each with its own `--out`:
   `smem-eval --split dev --config configs/raw_turns.yaml --backend llm --judge llm --ids "<ids>" --out evals/results_x/shard0`.
   Add `--store-budget 0.03` and `--ablate evict.policy=fifo` for the budget curve. Ingest state is cached
   per (question, write-side config) under `.cache/ingest`, so read-side changes re-run in minutes.
5. The held-out subset is `data/test200_seed0.json`; the runner takes its ids through `--ids`.

Config files may inherit with `extends: other.yaml` (keys override section by section). Run identity
excludes infrastructure fields (timeouts, devices, cache dirs); it includes everything that can change a
result.

## Results

Full tables and every variant tried: [docs/results.md](docs/results.md). Final configuration
`configs/raw_turns.yaml`: Qwen3-8B-AWQ extraction (v3-detail) → bge-m3 + BM25 → 2k tokens of packed
entries + 6k tokens of verbatim turns → gpt-4.1-mini; GPT-4o judge with the official prompt.

| | dev100 | test200 (held-out, 197 q) | combined 297 |
|---|---|---|---|
| store unbounded | 0.76 | **0.82** | **0.80** |
| oracle (gold evidence text → same reader) | 0.84 | | |

Accuracy vs store budget *B* (fraction of the history's tokens), combined 297 questions:

| B | 0.015 | 0.03 | 0.125 | 0.25 | ∞ |
|---|---|---|---|---|---|
| sieve + swap + Hawkes | **0.26** | **0.38** | **0.60** | **0.76** | **0.80** |
| FIFO | 0.22 | 0.27 | 0.60 | | |
| LRU (dev100 only) | 0.23 | | | | |

Below B = 0.06 the budget is spent entirely on extracted entries and no raw turn survives; from 0.125 every
entry fits and the remainder goes to raw turns, which is why swap and FIFO coincide there.

## Status and known gaps

- Everything above has been run with the real models; 107 tests cover the modules, the cache round trips,
  raw turns as budgeted entries and config inheritance.
- Consolidation is off: with the entailment direction fixed the summaries are accepted, but the write
  policy sees them as new candidates with ~0 gain over their still-stored members. It needs
  "replace the members" semantics before it can do anything.
- Small budgets are low because the coverage objective keeps what represents the stream, and the
  benchmark asks for needles; half of the evidence is gone at write time at B = 0.015. The write value
  function has not been worked on (see docs/process.md, "还没做的").
- multi-session counting and single-session-preference are the weak types on both splits.
- Extraction cache covers dev and the 197-question test subset only; the remaining 203 test questions
  would need ~7k more sessions extracted.
