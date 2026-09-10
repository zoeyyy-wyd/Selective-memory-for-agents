Technical plan v2 · Fall-recruiting agent systems project · 2026-09-10

# A Selective Memory System for Long-Conversation Agents

Give a long-conversation agent an external long-term memory whose capacity is bounded. Under a storage budget and a read budget, the system answers three questions: what is worth writing, what to forget when space runs out, and how much to read each time. Write, forget, and read are unified as one budgeted submodular coverage problem, with a Hawkes-process access prior and entailment-verified consolidation. Everything is a deterministic algorithm; nothing is trained. On LongMemEval, the deliverable is not a single score but the accuracy-vs-budget curve, plus three new metrics that directly measure writing, forgetting, and reading.

## 00 · Positioning

**The problem.** An agent has talked with a user across dozens of sessions and a hundred-thousand-plus tokens. The user asks "where was I living in March?" or "what allergy did I mention last time?" Stuffing the whole history into the context is expensive and does not scale. Existing memory systems (Mem0, Zep and the like) solve "retrieve from memory" but assume memory can grow without bound; they never answer "what should be remembered and what should be forgotten." Several 2026 surveys and benchmark reports say the same thing: *current evaluations almost only test reading; write policies are barely measured; forgetting and eviction are scored by no benchmark at all; token cost is generally ignored.* A separate analysis found that on weaker models, complex memory systems "fail silently" because of malformed write outputs and end up losing to simple ones.

**This project.** A memory system that explicitly accepts two budgets: a storage budget *B* (total memory tokens) and a read budget *R* (tokens injected into context per query). Under *B* it decides what to write, how to organize it, and what to forget; under *R* it decides what to read. Evaluation reports accuracy as a function of *B* and *R*, and uses the benchmark's evidence annotations to directly measure the write, survive, and inject steps.

**Scope.** No model is trained; no graph database; no multi-user isolation; no multimodality. The answering model is fixed; the project only changes the memory it sees.

> **In one sentence** — Treat memory as a finite resource to be managed rather than an append-only store, and make "did we write the right thing / forget the right thing" directly measurable for the first time.

## 01 · A Worked Example First

Walk one LongMemEval knowledge-update question through the system.

**History.** In session 3 (Feb 4) the user says "I just moved to Boston, renting near Kendall." In session 21 (May 19) the user says "moved to Seattle this week, company transfer." In between are thirty-odd sessions unrelated to moving—coding questions, recipes, drafting emails.

**Write.** Session 3 yields one episode ("user moved to Boston, renting near Kendall", Feb 4) and one fact `(user, city, Boston, valid_from=2/4)`. Session 21 yields `(user, city, Seattle)`—the write policy sees the same entity and attribute with a different value and does not overwrite: it sets the old fact's `valid_to` to May 19 and points its `superseded_by` at the new fact. From a recipe session, "user likes garlic" comes from a user preference, does not overlap existing entries, and its marginal coverage gain per token clears the threshold, so it is stored; "assistant suggested boiling water first" comes from the assistant and is already well covered by existing recipe entries, so its marginal gain is near zero and it is skipped.

**Forget.** The experiment sets *B* to 25% of the total history. By session 30 the store is full and new candidates must swap in: the algorithm finds the entries with the lowest marginal coverage per token—dozens of episodic entries already folded into the semantic layer, each nearly fully covered by its summary—and a new candidate replaces one whenever its gain exceeds theirs by a hysteresis factor. "User likes garlic" has not been mentioned in a while, but nothing else covers it, so its marginal gain is not low and it stays for now. The two city facts belong to the entity "user/city", which has been mentioned repeatedly and recently, so its Hawkes intensity is high, the weight is large, and the whole chain stays.

**Read.** Question: "which city does the user live in now?" Temporal parsing yields "now"; retrieval hits both city facts; following the chain to its tail gives Seattle. Had the question been "where did the user live in March?", the interval covering March is selected and the answer is Boston. Packing under *R* = 2k: the two city facts are the most relevant and redundant with each other; the budgeted greedy takes the chain tail first (largest gain), then the chain head because it still covers the "change over time" part of the relevant content; a third similar episodic entry has near-zero gain and is skipped, leaving budget for other candidates. The answering model replies Seattle and cites the entry ids.

**Measure.** The evidence sessions for this question are 3 and 21. Evaluation checks whether the two facts were written (write rate), whether they survived at *B* = 25% (survival rate), whether they were injected at *R* = 2k (injection rate), and whether the final answer is correct. Four numbers point respectively at write, forget, read, and answer; whichever link broke is visible at a glance.

## 02 · Benchmark and Data

| Item | Content |
|---|---|
| Main benchmark | **LongMemEval-S**: 500 questions, each paired with a timestamped chat history of ~115k tokens across ~40 sessions. Six types: single-session (user / assistant / preference), multi-session reasoning, temporal reasoning, knowledge update, abstention (should say "I don't know" when the history has no answer). MIT license, HuggingFace `xiaowu0162/longmemeval-cleaned` |
| Evidence labels | Each question provides `answer_session_ids` and turn-level `has_answer`. **This is what makes writing and forgetting directly measurable** |
| History construction | Evidence sessions interleaved with filler sessions sampled from ShareGPT / UltraChat. Filler sessions are shared across questions, so session-level extraction can be cached once and reused by every configuration |
| Official judge | GPT-4o; `evaluate_qa.py` emits an `autoeval_label` per question |
| Split | Stratified by type: 100 questions as **dev** (tune thresholds, weights, prompts), 400 as **test** (run only at the end). No parameter is tuned on test |
| State of the art | Retrieval-heavy systems self-report 90%+ on -S (setups differ; contested). This project does not compete on that number: the x-axis is budget, and what matters is the shape of the curve and the cost |
| Optional extensions | LongMemEval-M (~500 sessions) for heavier storage pressure; a subset of BEAM-1M (ICLR 2026) for scaling |

#### Baselines

Four, all with the same answering model and the same judge. **Oracle**: inject only the annotated evidence sessions—the answering model's ceiling; everything else sits below it. **Full context**: feed the entire history to the answering model, run only on the 100 dev questions to control cost. **Naive RAG**: chunk sessions, bge-m3, top-k. **Mem0 open-source**: deployed locally, extraction model pointed at local Qwen.

## 03 · Architecture

```
WRITE PATH (sessions arrive in time order; the budget check runs after every session)

  session ─▶ extract (local model, schema-constrained decoding) ─▶ candidate episodes + facts
                    │
                    ▼
            write policy: novelty · specificity · source → add / merge / skip
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   Episodic store         Semantic store
   events with timestamps, (entity, attribute, value) + validity chains,
   pointing back to turns  pointing back to episodes
          │                    ▲
          └── consolidate ─────┘   every N sessions: cluster within a time window → compress into semantic entries
                    │
                    ▼
            evict: total > B → evict by utility until back under budget

READ PATH (a question arrives)

  question ─▶ parse temporal constraint ─▶ rewrite into retrieval queries
           ─▶ hybrid retrieval BM25 ⊕ bge-m3 (both stores) ─▶ follow validity chains to the right version
           ─▶ pack: maximize relevance, penalize redundancy within budget R (greedy MMR)
           ─▶ inject (with ids, timestamps, source type; time-ordered)
           ─▶ answering model: answer only from injected content, cite ids, abstain if no support ─▶ judge
```

Five modules, each with a switch: write policy, validity chains, consolidation, eviction policy, packing policy. Each switch is one ablation. The write path is **online**—sessions enter in time order and eviction happens before questions arrive, so the storage budget is a real constraint rather than an after-the-fact truncation.

## 04 · Data Model

```
class Episode(BaseModel):
    id: str
    ts: datetime                      # session timestamp (absolute)
    session_id: str; turn_idx: int    # pointer back to the source turn; used by eval and UI
    speaker: Literal["user", "assistant"]
    text: str                         # one-sentence event, ≤ 60 tokens
    entities: list[str]
    embedding: list[float]; tokens: int
    consolidated: bool = False        # already folded into the semantic layer
    access_count: int = 0; last_access: Optional[datetime] = None

class Fact(BaseModel):
    id: str
    entity: str; attribute: str; value: str     # ("user", "city", "Seattle")
    valid_from: datetime
    valid_to: Optional[datetime] = None
    superseded_by: Optional[str] = None         # knowledge update: a chain, never a delete
    sources: list[str]                          # Episode ids
    kind: Literal["stated", "preference", "inferred"]
    embedding: list[float]; tokens: int
    access_count: int = 0; last_access: Optional[datetime] = None

class Budget(BaseModel):
    store_tokens: int          # B
    read_tokens: int           # R
    consolidate_every: int     # consolidate every N sessions
```

Storage: SQLite for metadata and chains, FAISS for vectors, a BM25 index in memory. A single history has entries in the low thousands; no vector database is needed.

## 05 · Algorithmic Core: One Objective, Two Budgets

Write, forget, and read do not need three separate heuristics. They are instances of the same optimization problem at two scales: select a subset under a token budget to maximize coverage of some set. The coverage function is submodular, so both greedy and streaming algorithms come with approximation guarantees.

#### Coverage function

```
# facility-location form, monotone submodular
f_W(S) = Σ_{h ∈ H} w_h · max_{s ∈ S} sim(h, s)

H: the set to be covered            S: the selected subset (under a token budget)
w_h: importance weight of h         sim: embedding cosine similarity (computed on FAISS)
```

**Storage scale**: H is every candidate extracted so far, S is the store, the budget is B. Weight `w_h = spec(h) · src(h) · λ_e(t)`—specificity, source, and the *predicted access intensity* of the entry's entity (defined below). The goal is to use B tokens to cover the parts of the history that are important and likely to be asked about again.

**Read scale**: H is the retrieval candidate set for this query, S is the subset injected into context, the budget is R, weight `w_h = rel(q, h)`. The goal is to use R tokens to cover the content relevant to the question without repetition. MMR is a special case of this objective (the greedy with a fixed λ).

#### Predicted access intensity: a Hawkes process

```
λ_e(t) = μ_e + Σ_{t_i < t} α · exp(−β (t − t_i))     # self-exciting process for entity e

t_i: past times at which entity e was mentioned or hit by retrieval
μ_e: base intensity, estimated from the entity's mention frequency in the history
α, β: excitation magnitude and decay, fit on dev by maximum likelihood or grid search
```

This turns "recency × frequency" from two hand-picked weights into a point process with a parametric form: entities mentioned repeatedly and recently have high intensity; entities not mentioned for a long time decay exponentially but never to zero. An entry's utility takes the maximum λ over its entity set. This is the eviction-time estimate of "will this be asked about again?"

## 06 · Write and Evict: Streaming Submodular Selection

Sessions stream in over time, candidates arrive one at a time, and the store has a capacity cap. This is streaming submodular maximization under a knapsack constraint; writing and eviction are two branches of the same algorithm.

#### Extraction

Qwen3-8B (AWQ) served by vLLM on an L4; for each session it outputs `{episodes, facts}` with schema-constrained decoding via xgrammar or outlines. Relative times ("last Wednesday") are normalized to absolute dates at extraction time using the session timestamp. Results are cached by `session_id`; the whole project runs extraction once. The write error rate (fraction of outputs violating the schema) is reported as a metric.

#### Dedup and update (deterministic, before selection)

Same entity, attribute, and value → merge (union the sources). Same entity and attribute, different value → update (validity chain, section 07). Only truly new candidates enter the selection below.

#### Streaming selection with swaps

```
on_arrive(x):                                   # for each new candidate
    g = Δf(x | S) / cost(x)                     # marginal coverage gain per token
    if used + cost(x) ≤ B and g ≥ θ:            # room available: threshold admission
        S.add(x)
    elif used + cost(x) > B:                    # full: consider a swap
        y = argmin_{y ∈ S, evictable} Δf(y | S∖y) / cost(y)
        if Δf(x | S∖y) / cost(x) ≥ (1 + γ) · Δf(y | S∖y) / cost(y):
            S.swap(y → x)
    else: skip(x)

evictable: consolidated episodic entries first; a validity chain is scored as a whole and swapped as one unit
```

θ is not one number but a geometric sequence of thresholds, each maintaining its own candidate store (the SieveStreaming construction); the store with the largest f wins at the end. A single-threshold version serves as a cheap control. γ is a hysteresis coefficient that prevents oscillation. For monotone submodular objectives this family carries constant-factor guarantees (roughly 1/2 − ε for threshold streaming; slightly lower for the swap variant, which supports deletion). The "novelty filter" at write time and the "utility score" at eviction time both become special cases: novelty is Δf, and utility is Δf / cost with the Hawkes weight folded in.

Complexity: each candidate needs one max-similarity query against S, O(log|S|) on FAISS; the incremental update of Δf touches only the h whose max was changed by x, tracked with a `best_cover[h]` array. A full 115k-token history runs in seconds on CPU.

#### Controls

Write side: `all` (store everything), `novelty_threshold` (the v1 heuristic), `sieve` (this design). Eviction side: `fifo`, `lru`, `random`, `utility_heuristic` (the v1 weighted sum), `swap` (this design), `swap_no_hawkes` (weight without λ_e). Metrics remain evidence write rate and evidence survival rate as functions of B.

## 07 · Knowledge Updates and Time

When a new value appears for the same `(entity, attribute)`: set the old fact's `valid_to` to the new session time and its `superseded_by` to the new fact; the new fact's `valid_from` is that time. Chains are time-ordered and intermediate nodes are never deleted; at eviction time a chain is one unit.

**Attribute volatility.** For each `(entity, attribute)` maintain the update count and time span and estimate a volatility `v = updates / span`. It decides how much to trust the chain tail for "now": for high-volatility attributes (current city, current project) "now" is the tail; for low-volatility attributes (birthday, allergen) a tail that contradicts the head and comes from the assistant or inference is flagged as suspect, and both ends are injected so the answering model can judge. This is a lightweight belief-revision rule aimed at the boundary between the benchmark's "knowledge update" and "contradiction" cases.

**Temporal constraint parsing and interval algebra.** A question's temporal constraint is parsed into an interval `[t₁, t₂]` ("now" is `[t_now, t_now]`, "March" is the whole month, "last time" is just before the latest update); fact validity is also an interval; the chain node whose interval intersects is selected. Questions about the change itself ("how many times did I move?") return the whole chain. The `no_validity_chain` ablation (overwrite) is expected to drop noticeably on the knowledge-update and temporal-reasoning types.

## 08 · Consolidation: Redundancy-Triggered, Entailment-Verified

Consolidation is not run on a fixed interval; it is triggered by **redundancy**: when a cluster exists in the episodic store whose members' summed marginal coverage gains are far below the member count (they cover each other), it is worth compressing. Concretely: after each session, threshold-cluster the new entries; a cluster fires when `Σ Δf(h | S∖h) / |cluster|` falls below a threshold.

Compression is done by the local model: cluster → 1–3 semantic facts. **The acceptance condition is entailment verification**: a small local NLI model (DeBERTa-class) checks whether every member episode is entailed by the generated summary; summaries whose entailment rate falls below a threshold are discarded and the cluster is left as is. Consolidation is therefore "near-lossless" and the loss is measurable—report the fraction of rejected summaries and the change in evidence survival rate after consolidation. Member episodes are tagged `consolidated` and are first out at swap time.

Controls: `fixed_interval` (every N sessions), `no_consolidation`, `no_nli_check`.

## 09 · Read: Multi-Hop Retrieval and Budgeted Packing

#### Retrieval

Parse the temporal constraint → rewrite into 1–2 queries → BM25 and bge-m3 take 30 each, fused with RRF → follow validity chains to the version whose interval intersects.

**Multi-hop.** Evidence for multi-session reasoning questions is scattered across sessions, and a single retrieval often hits only one end. Do two hops: extract the entity set E₁ from first-hop candidates, take entries co-occurring with E₁ from the fact store's entity index as second-hop candidates, and merge. The budget is split between hops adaptively according to the first hop's relevance distribution (a confident first hop leaves less for the second). Control: `single_hop`.

#### Packing

```
max  g(T) = Σ_{h ∈ cand} rel(q, h) · max_{t ∈ T} sim(h, t)     s.t.  Σ_{t ∈ T} tok(t) ≤ R

greedy: at each step pick argmax Δg(t | T) / tok(t); take the better of that and the best single element
(Khuller–Moss–Naor budgeted maximum coverage, (1 − 1/e)/2 guarantee; CELF lazy evaluation for speed)
```

Same function family as the storage scale, with query relevance as the weight and retrieval candidates as the set. MMR is the control (the fixed-λ special case) and top-k is the lower bound. Each injected entry carries an id, a timestamp, and a source type, ordered by time.

#### Abstention

Abstain only when both conditions hold: the highest relevance after packing is below `τ_abs`, and the injected content contains none of the question's entities. `τ_abs` is set on dev to maximize joint F1 across abstention and non-abstention types; a calibration curve is reported. The answering model's instruction: answer only from injected content, cite entry ids, say "I don't know" when unsupported.

Report accuracy and actual tokens/query at R ∈ {1k, 2k, 4k, 8k}, on the same chart as Mem0's self-reported 6.8k tokens/query.

## 10 · Evaluation

| Metric | Definition | Measures |
|---|---|---|
| Accuracy (six types separately) | Official judge script; dev judged by gpt-4.1-mini, final test by GPT-4o | overall |
| Evidence write rate | Fraction of entries extracted from annotated evidence sessions that were written to the store | write |
| Evidence survival rate @ B | Fraction of evidence entries still in the store under budget B | forget |
| Evidence injection rate @ R | Fraction of evidence entries packed into context under budget R | read |
| Write error rate | Fraction of extraction outputs violating the schema (constrained vs. unconstrained) | robustness |
| Abstention | Correct abstention rate on unanswerable questions; false abstention rate on answerable ones | trustworthiness |
| Cost | Store size, tokens/query, extraction and consolidation call counts | cost |

The first three metrics decompose "answered wrong" into "never written / evicted / not read / read but answered wrong", each pointing at one module. This diagnostic capability is something no existing benchmark provides and is what separates this project from Mem0-style systems.

#### Three main figures

Figure 1: x-axis storage budget B (100% / 50% / 25% / 12.5% of history tokens), y-axis accuracy; lines: this system, FIFO, LRU, random, naive RAG (no eviction, horizontal), Oracle (horizontal ceiling). Figure 2: x-axis read budget R; lines: budgeted greedy vs. MMR vs. top-k, with the Mem0 point. Figure 3: broken down by question type, showing the validity-chain ablation on the knowledge-update and temporal-reasoning types.

#### Ablations

Write `all | novelty_threshold | sieve` · Evict `fifo | lru | random | utility_heuristic | swap | swap_no_hawkes` · `no_validity_chain` · Consolidation `fixed_interval | no_consolidation | no_nli_check` · Retrieval `single_hop | two_hop` · Packing `topk | mmr | budgeted_greedy` · `no_constrained_decoding`. Each group corresponds to one design decision in sections 05–09; the central comparison is "submodular + Hawkes" against "v1 heuristics"—whether the algorithm with a guarantee actually wins on the data.

#### Reproducibility

Extraction cache, retrieval results, injected content, answers, and judge outputs are all persisted with a config hash; the judge runs at a fixed temperature; every number carries a bootstrap 95% CI (resampled by question); differences smaller than the CI are not written into conclusions.

## 11 · Compute and Cost

| Stage | Where | Volume | Cost |
|---|---|---|---|
| Extraction (one-time) | L4, Qwen3-8B AWQ, vLLM + constrained decoding | deduplicated sessions × ~3k tokens; worst case 500 × 40 = 20k sessions, 60M tokens | $0; worst case one overnight run; far less once filler sharing is accounted for |
| Embeddings, consolidation, query rewriting | L4, bge-m3 and the same model | small | $0 |
| Answering | gpt-4.1-mini (fixed) | 500 questions × ~5k tokens × ~10 configurations | ~$10 |
| Judge | dev: gpt-4.1-mini; final test: GPT-4o | 500 × ~1k × number of configurations | dev ~$2; final ~$1.5 per configuration |
| Full-context baseline | gpt-4.1-mini, 100 dev questions | 100 × 115k | ~$5 |

Around $30 in total. The answering model can be swapped for Qwen3-8B on the L4 to bring cost to zero, but a weak answering model can wash out the differences between memory systems—compare the two on 50 questions in week one before deciding.

## 12 · Engineering

```
selective-memory/
├── pyproject.toml
├── src/smem/
│   ├── schemas.py                # Episode / Fact / Budget
│   ├── extract.py                # extraction, constrained decoding, time normalization, session-level cache
│   ├── write.py                  # novelty / specificity / source → add / merge / update / skip
│   ├── store.py                  # SQLite + FAISS + BM25; validity-chain operations
│   ├── consolidate.py
│   ├── evict.py                  # utility, FIFO/LRU/random controls, Thompson weight tuning
│   ├── read.py                   # temporal parsing, rewriting, hybrid retrieval, chain resolution, MMR packing
│   ├── answer.py                 # injection format, id-citation check, abstention
│   └── llm.py · config.py
├── baselines/{oracle,full_context,naive_rag,mem0_oss}.py
├── evals/
│   ├── longmemeval/              # loading, dev/test split, official judge wrapper
│   ├── metrics.py                # write / survival / injection rates, abstention, cost, CI
│   ├── run.py                    # --split --config --store-budget --read-budget --ablate
│   ├── ablations.yaml
│   └── results/                  # one directory per run, named by config hash
├── demo/cli.py                   # feed a conversation, ask questions, print what was injected and what was evicted
├── tests/                        # validity chains, knapsack packing, eviction order, schema, temporal parsing
└── README.md                     # three figures + ablation table + the worked example
```

Dependencies: `vllm`, `xgrammar` or `outlines`, `sentence-transformers` (bge-m3), `faiss-cpu`, `rank-bm25`, `pydantic`, `openai`, `datasets`, `dateparser`, `mem0ai` (baseline), `pytest`. All intermediate artifacts are written to disk; no tracing platform is needed.

## 13 · Build Order

1. **Data, split, evaluation skeleton** — Download LongMemEval-S, stratify dev/test, count deduplicated sessions, wrap the official judge, implement the three evidence metrics and CIs. Check: the Oracle configuration runs and yields the answering model's ceiling.
2. **Extraction and cache** — Serve Qwen3-8B on the L4 with constrained decoding, run every session, cache. Check: write error rate is 0; manually inspect extraction quality and time normalization on 30 sampled sessions.
3. **Baselines** — Naive RAG on everything; full context on the 100 dev questions; compare gpt-4.1-mini and local Qwen3-8B as the answering model on 50 questions and pick one. Check: baseline rows of the main table are filled.
4. **Minimal system** — Store everything + hybrid retrieval + top-k injection, B unbounded. Check: not below naive RAG. This is the reference point for every later module.
5. **Validity chains + temporal parsing** — Check: knowledge-update and temporal-reasoning types improve over step 4.
6. **Submodular write + budgeted packing** — Coverage function with incremental maintenance, threshold streaming write, budgeted greedy packing. Check: the write-fraction vs. evidence-write-rate curve has a knee, and `sieve` beats `novelty_threshold` at equal write fraction; dropping R from 8k to 2k hurts budgeted greedy less than top-k and MMR.
7. **Swap eviction + Hawkes** — Swap rule, chains as units, Hawkes intensity estimation and parameter fitting on dev. Check: at B = 50% / 25%, `swap` is above FIFO / LRU / heuristic utility; the `swap_no_hawkes` control shows what the access prior is worth. This is Figure 1.
8. **Redundancy-triggered consolidation + two-hop retrieval + Mem0 baseline** — NLI entailment check, trigger threshold, entity-index two-hop. Check: rejected-summary fraction and evidence survival before/after consolidation; multi-session type improves under `two_hop`; Mem0's score and tokens/query under the same answering model and judge.
9. **Finalize** — Run the final configuration and all baselines on the 400 test questions with the GPT-4o judge; three figures; README with the section 01 example; CLI demo.

Three weeks: steps 1–3 in week one, 4–6 in week two, 7–9 in week three. End-of-week-one checkpoint: Oracle ceiling, naive-RAG baseline, answering model chosen—all three in place before entering week two. If time runs short, cut consolidation and two-hop from step 8 first; keep the Mem0 baseline, which is the direct evidence for "how this beats an off-the-shelf system."

> **Risk** — The main risk is effect size: if at B = 25% this system is only two or three points above LRU and inside the CI, Figure 1 does not hold. Mitigation: run an extreme stress test at B = 12.5% on dev first and confirm the gap exists before polishing; if no gap appears, shift the focus to validity chains and the read budget, whose gains are more certain.

## 14 · Resume and Interview

- **A Selective Memory System for Long-Conversation Agents**: explicitly accepts storage and read token budgets and unifies write, eviction, and read as knapsack-constrained submodular coverage maximization—write and eviction via a threshold-streaming + swap algorithm (SieveStreaming variant) selected online under the storage budget, with entry weights from a Hawkes-process estimate of entity access intensity; read via budgeted maximum-coverage greedy (Khuller–Moss–Naor) packed under the read budget; knowledge updates represented as validity chains with attribute-volatility belief revision; consolidation triggered by redundancy and verified by NLI entailment; two-hop entity-graph retrieval for multi-session questions.
- On LongMemEval-S (500 questions, 115k-token histories), compared against Oracle, full context, naive RAG, and Mem0 under the same answering model and judge: at 25% storage budget accuracy drops only X points (LRU drops Y), with tokens/query at Z% of Mem0's; proposes evidence write / survival / injection rates as three metrics that directly measure writing, forgetting, and reading, quantified across seven ablation groups.

Relation to existing projects: the DPO research is the data side (how to sample across sources), Agentic Video QA is the training side (RL learns to use tools within a budget), and this is the memory side (deciding what to remember and forget within a budget). All three are about making a model do the right thing under limited resources.

**"Why submodular instead of just learning a scorer?"** Submodular coverage gives an approximation guarantee, and write, forget, and read share one objective, so there is only one thing in the system to explain; a learned scorer is the next step and can directly replace the weight w_h.

**"What does Hawkes assume here, and when does it fail?"** It assumes entity access is self-exciting: what was just mentioned is more likely to be mentioned again. It fails on one-off events ("book me a ticket"), which is why the weight also multiplies in specificity and source rather than relying on intensity alone.

**"Can consolidation compress the evidence away?"** Summaries that fail the NLI entailment check are not accepted; the rejected fraction and evidence survival before and after consolidation are reported, so the loss is measurable.

**"Why not train a write policy?"** First show how much headroom a deterministic policy has on the budget curve; training is the next step, and the reward signal (evidence labels) is already defined in the evaluation and can plug directly into GRPO.

**"How do validity chains differ from overwriting?"** Temporal-reasoning questions ask "where in March"; after an overwrite that is unanswerable. Figure 3 is the ablation.

**"What's the essential difference from Mem0?"** Mem0 pursues accuracy at unbounded cost and has no notion of a storage budget; this system targets the accuracy–cost frontier and directly measures writing and forgetting.

**"Isn't LongMemEval-S nearly saturated?"** With unlimited budget, yes; this project's x-axis is budget, and saturation does not change the shape of the curve.

**"Evidence write rate is 100% but the answer is wrong—what does that mean?"** The problem is in reading or answering, not writing—which is exactly why the three metrics exist: to attribute the error to a module.

Sources:

- [LongMemEval (data, judge scripts, evidence annotations)](https://github.com/xiaowu0162/LongMemEval)
- [Anatomy of Agentic Memory (2026-02): silent failure, simple systems win on weak models, open problems](https://arxiv.org/html/2602.19320v1)
- [Mem0: AI memory benchmarks in 2026 — the write step is barely measured, cost is ignored](https://mem0.ai/blog/ai-memory-benchmarks-in-2026) · [Cognee: AI memory benchmarks guide (BEAM, MemoryAgentBench, etc.)](https://www.cognee.ai/ai-memory-benchmarks)
- [LongMemEval-V2 (2026-05)](https://arxiv.org/html/2605.12493v1) · [Anthropic: context management and the memory tool](https://www.anthropic.com/news/context-management)
- Algorithm references: Badanidiyuru et al., *Streaming Submodular Maximization* (KDD 2014, SieveStreaming); Khuller, Moss & Naor, *The Budgeted Maximum Coverage Problem* (1999); Leskovec et al., CELF (KDD 2007); Carbonell & Goldstein, MMR (SIGIR 1998); Hawkes, *Spectra of some self-exciting point processes* (1971)
