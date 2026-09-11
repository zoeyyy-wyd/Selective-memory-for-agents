# A Selective Memory System for Long-Conversation Agents

An external long-term memory for a long-conversation agent whose capacity is **bounded**. Under a
storage budget *B* (total memory tokens) and a read budget *R* (tokens injected per query) the system
decides what is worth writing, what to forget when space runs out, and how much to read each time.
Write, forget and read are one budgeted submodular coverage problem, with a Hawkes-process access prior
and entailment-verified consolidation. Nothing is trained. The deliverable on LongMemEval is the
accuracy-vs-budget curve plus three metrics that measure writing, forgetting and reading directly.

The technical plan is in [selective-memory-for-long-conversation-agents.md](selective-memory-for-long-conversation-agents.md).
This repository implements it end to end; the numbers below are placeholders until the GPU/API runs
in section "Running the real experiments" have been done.

## Setup

```bash
conda env create -f environment.yml      # python 3.11, faiss-cpu, torch, sentence-transformers
conda activate smem                      # (the env already has `pip install -e .[dev]` applied)
pytest                                   # 66 tests, all offline, ~1 min
```

Two backends exist for every model-dependent step so that the whole pipeline runs without a GPU or an
API key:

| step | `--backend offline` (tests, demo, dry runs) | `--backend llm` (real experiments) |
|---|---|---|
| extraction | rule-based `HeuristicExtractor` | Qwen3-8B via vLLM, schema-constrained (`guided_json`) |
| embeddings | feature-hashing `HashEmbedder` | `BAAI/bge-m3` via sentence-transformers, cached on disk |
| consolidation summary | medoid episode | same local model, JSON schema |
| entailment check | lexical overlap | `cross-encoder/nli-deberta-v3-base` |
| answering | extractive stand-in | `gpt-4.1-mini` (any OpenAI-compatible model) |
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
  ablations.yaml  one group per design decision + the two budget sweeps
baselines/        oracle, full_context, naive_rag, mem0_oss (§02)
configs/          default.yaml (real models), offline.yaml (no models)
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

**Weights.** `w_h = spec(h) · src(h) · λ_e(t)` with `λ_e` the Hawkes intensity of the entry's hottest
key: facts belong to their `entity/attribute` process (e.g. `user/city`), episodes to their named
entities; mentions and retrieval hits are the events. `swap_no_hawkes` sets `λ = 1`.

**Read (`read.py`).** Temporal constraint → 1–2 queries → BM25 ⊕ dense fused with RRF → every fact hit
is replaced by the chain node(s) its interval intersects ("now" = tail, subject to the volatility rule;
change questions = whole chain) → optional entity two-hop with an adaptive budget share → packing under
*R* with KMN budgeted greedy + CELF (MMR and top-k are the controls) → abstain when the top relevance is
below `τ_abs` *and* no question term appears in the packed content.

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

### Running the real experiments

1. Serve the extraction model on the GPU box:
   `vllm serve Qwen/Qwen3-8B-AWQ --guided-decoding-backend xgrammar --port 8000`
   and point `models.extract_base_url` at it (SSH tunnel is fine; extraction is cached per session).
2. `export OPENAI_API_KEY=...`; the answering and judge models are set in `configs/default.yaml`.
3. Build order as in plan §13: oracle / full-context / naive-RAG baselines on dev, then the system with
   *B* unbounded, then the `evict` sweep at *B* ∈ {0.125, 0.06, 0.03, 0.015} (Figure 1), the `packing` sweep at
   *R* ∈ {1k, 2k, 4k, 8k} (Figure 2), and `validity_chain` by question type (Figure 3). Test split only at
   the end, with `--judge llm` and `models.judge_model: gpt-4o`.
4. Fit the Hawkes parameters on dev with `HawkesIntensity.fit()` over the dev histories and put the
   result in the config; set `read.tau_abs` from the dev abstention calibration.

## Results

| configuration | accuracy @ B=100% | @ 50% | @ 25% | @ 12.5% | tokens / query |
|---|---|---|---|---|---|
| this system (sieve + swap + Hawkes) | — | — | — | — | — |
| LRU / FIFO / random / utility heuristic | — | — | — | — | — |
| naive RAG (no eviction) | — | — | — | — | — |
| Oracle | — | — | — | — | — |

To be filled from `evals/results/*/summary.json` after the runs above. Offline dry runs produce the
evidence metrics but not meaningful accuracy (hash embeddings, extractive answering).

## Status and known gaps

- Implemented and tested offline: every module, every ablation switch, all four baselines' plumbing,
  the metrics with CIs, the runner, the demo.
- Not yet run: anything needing the GPU or an API key (extraction with Qwen3-8B, bge-m3 embeddings,
  the DeBERTa NLI check, gpt-4.1-mini answering, the LLM judge, Mem0). Those code paths exist but have
  only been exercised through their offline stand-ins and scripted-LLM tests.
- The judge prompts are a port of upstream `evaluate_qa.py`; diff against the LongMemEval repository
  before the final test run.
- Key drift has only been exercised with the heuristic extractor, whose keys come from a fixed table.
  The first thing to inspect after the real dev extraction is `key_drift` in `summary.json` / `records.jsonl`.
- `HashEmbedder` similarity is lexical, so offline relevance/abstention behaviour is only indicative;
  `read.tau_abs` and the Hawkes parameters are meant to be tuned on dev with the real models.
