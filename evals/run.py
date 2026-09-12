"""Evaluation runner.

  smem-eval --split dev --config configs/default.yaml --backend llm --store-budget 0.25 --read-budget 2000
  smem-eval --split dev --config configs/offline.yaml --backend offline --judge exact --limit 20
  smem-eval --baseline naive_rag --split dev ...
  smem-eval --ablation-group evict --split dev --store-budget 0.25

Every run writes evals/results/<config hash>/{config.yaml, records.jsonl, summary.json}. The config
hash covers the system config, budgets, backend, split and baseline, so a re-run with the same
arguments lands in the same directory and skips questions that already have a record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from evals.longmemeval.data import (
    LMEQuestion,
    count_unique_sessions,
    download_longmemeval,
    load_longmemeval,
)
from evals.longmemeval.judge import ExactMatchJudge, Judge, LLMJudge, NoJudge
from evals.longmemeval.split import load_split, save_split, stratified_split
from evals.metrics import QuestionRecord, aggregate, evidence_counts, format_summary
from smem.config import SystemConfig, load_dotenv, parse_override
from smem.system import SelectiveMemory, build_llm, build_system


def resolve_store_budget(value: float | None, history_tokens: int) -> int:
    if value is None:
        return 10**9
    if value <= 1.0:
        return max(int(value * history_tokens), 1)
    return int(value)


def run_question(cfg: SystemConfig, q: LMEQuestion, backend: str, judge: Judge, store_budget: float | None
                 ) -> QuestionRecord:
    history = q.history_tokens
    q_cfg = cfg.with_overrides({"budget.store_tokens": resolve_store_budget(store_budget, history)})
    mem: SelectiveMemory = build_system(q_cfg, backend)
    try:
        mem.ingest(q.sessions)
        result = mem.ask(q.question, now=q.question_date)
        counts = evidence_counts(mem, q, result.injected_ids)
        correct = None if isinstance(judge, NoJudge) else judge.judge(q, result.answer)
        stats = mem.stats()
        return QuestionRecord(
            question_id=q.question_id, question_type=q.question_type, is_abstention=q.is_abstention,
            question=q.question, gold=q.answer, answer=result.answer, abstained=result.abstained, correct=correct,
            injected_tokens=result.read.tokens, injected_entries=len(result.read.packed),
            store_tokens=stats["store_tokens"], store_entries=stats["store_entries"], history_tokens=history,
            invalid_citations=len(result.invalid_citations), config_hash=q_cfg.config_hash(),
            extra={"stats": stats, "injected_ids": sorted(result.injected_ids), "constraint": result.read.constraint.mode,
                   "queries": result.read.queries, "store_budget": q_cfg.budget.store_tokens},
            **counts,
        )
    finally:
        mem.close()


def run_baseline(name: str, cfg: SystemConfig, q: LMEQuestion, backend: str, judge: Judge) -> QuestionRecord:
    from baselines import build_baseline

    baseline = build_baseline(name, cfg, backend)
    out = baseline.answer_question(q)
    correct = None if isinstance(judge, NoJudge) else judge.judge(q, out.answer)
    abstained = out.answer.strip().lower().startswith("i don't know")
    return QuestionRecord(
        question_id=q.question_id, question_type=q.question_type, is_abstention=q.is_abstention, question=q.question,
        gold=q.answer, answer=out.answer, abstained=abstained, correct=correct, injected_tokens=out.injected_tokens,
        injected_entries=out.injected_entries, history_tokens=q.history_tokens, config_hash=cfg.config_hash(),
        extra={"baseline": name, **out.extra},
    )


def build_judge(name: str, cfg: SystemConfig) -> Judge:
    if name == "none":
        return NoJudge()
    if name == "exact":
        return ExactMatchJudge()
    if name == "llm":
        return LLMJudge(build_llm(cfg.models.judge_model, cfg.models.judge_base_url, cfg,
                                  provider=cfg.models.judge_provider, api_key_env=cfg.models.judge_api_key_env),
                        max_tokens=cfg.models.judge_max_tokens)
    raise ValueError(f"unknown judge {name}")


def load_questions(args) -> list[LMEQuestion]:
    path = Path(args.data) if args.data else Path("data") / "longmemeval_s_cleaned.json"
    if not path.exists():
        path = download_longmemeval("s", path.parent)
    questions = load_longmemeval(path)
    split_path = path.parent / f"split_dev{args.dev_size}_seed{args.seed}.json"
    if split_path.exists():
        dev, test = load_split(questions, split_path)
    else:
        dev, test = stratified_split(questions, args.dev_size, args.seed)
        save_split(dev, test, split_path)
    chosen = {"dev": dev, "test": test, "all": questions}[args.split]
    if args.types:
        wanted = set(args.types.split(","))
        chosen = [q for q in chosen if q.question_type in wanted]
    if args.ids:
        wanted_ids = set(args.ids.split(","))
        chosen = [q for q in chosen if q.question_id in wanted_ids]
    if args.limit:
        chosen = chosen[: args.limit]
    return chosen


def run_id(cfg: SystemConfig, args, overrides: dict[str, Any]) -> str:
    payload = {"cfg": cfg.config_hash(), "backend": args.backend, "split": args.split, "baseline": args.baseline,
               "store_budget": args.store_budget, "read_budget": args.read_budget, "judge": args.judge,
               "types": args.types, "overrides": overrides}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def execute(cfg: SystemConfig, args, overrides: dict[str, Any], questions: list[LMEQuestion]) -> dict[str, Any]:
    rid = run_id(cfg, args, overrides)
    out_dir = Path(args.out) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump({"config": cfg.model_dump(mode="json"), "args": vars(args),
                                                          "overrides": overrides}, sort_keys=False))
    records_path = out_dir / "records.jsonl"
    done: dict[str, QuestionRecord] = {}
    if records_path.exists() and not args.rerun:
        for line in records_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["question_id"]] = QuestionRecord(**d)
    judge = build_judge(args.judge, cfg)
    records: list[QuestionRecord] = []
    t0 = time.time()
    with open(records_path, "a" if done and not args.rerun else "w") as f:
        for i, q in enumerate(questions):
            if q.question_id in done:
                records.append(done[q.question_id])
                continue
            if args.baseline:
                rec = run_baseline(args.baseline, cfg, q, args.backend, judge)
            else:
                rec = run_question(cfg, q, args.backend, judge, args.store_budget)
            records.append(rec)
            f.write(json.dumps(rec.as_dict(), default=str) + "\n")
            f.flush()
            if not args.quiet:
                mark = "?" if rec.correct is None else ("✓" if rec.correct else "✗")
                print(f"[{i + 1}/{len(questions)}] {mark} {rec.question_type:26s} ev {rec.n_written}/{rec.n_survived}/"
                      f"{rec.n_injected}/{rec.n_evidence}  tok {rec.injected_tokens:5d}  {rec.answer[:70]!r}", flush=True)
    summary = aggregate(records, n_boot=args.n_boot, seed=args.seed)
    summary["run_id"] = rid
    summary["elapsed_s"] = time.time() - t0
    summary["config_hash"] = cfg.config_hash()
    summary["overrides"] = overrides
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n== run {rid} ({cfg.name}{' ' + args.baseline if args.baseline else ''}) -> {out_dir}")
    print(format_summary(summary))
    health = judge.health() if hasattr(judge, "health") else None
    if health:
        summary["judge_health"] = health
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"\nWARNING  {health}")
    if summary["n_judged"] < summary["n"]:
        print(f"WARNING  accuracy is over {summary['n_judged']}/{summary['n']} questions; the rest are unjudged.")
    sub = getattr(getattr(judge, "llm", None), "substitutions", dict)()
    if sub:
        print(f"WARNING  the judge gateway served {sub} instead of {cfg.models.judge_model}; "
              "runs graded by different models are not comparable.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LongMemEval evaluation of the selective memory system")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend", choices=["offline", "llm"], default="offline")
    p.add_argument("--judge", choices=["none", "exact", "llm"], default="none")
    p.add_argument("--store-budget", type=float, default=None,
                   help="≤1: fraction of the history's tokens; >1: absolute tokens; omitted: unbounded")
    p.add_argument("--read-budget", type=int, default=None)
    p.add_argument("--ablate", action="append", default=[], help="dotted override, e.g. write.policy=all")
    p.add_argument("--ablation-group", default=None, help="run every row of a group in evals/ablations.yaml")
    p.add_argument("--ablations-file", default="evals/ablations.yaml")
    p.add_argument("--baseline", choices=["oracle", "full_context", "naive_rag", "mem0_oss"], default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--dev-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--types", default=None, help="comma-separated question types to keep")
    p.add_argument("--ids", default=None, help="comma-separated question ids to keep")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default="evals/results")
    p.add_argument("--rerun", action="store_true", help="ignore existing records for this run id")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--stats-only", action="store_true", help="print dataset statistics and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    questions = load_questions(args)
    if args.stats_only:
        uniq, total = count_unique_sessions(questions)
        print(f"{len(questions)} questions; {total} session slots; {uniq} unique sessions; "
              f"mean history tokens {sum(q.history_tokens for q in questions) / max(len(questions), 1):.0f}")
        return 0
    overrides: dict[str, Any] = {}
    for item in args.ablate:
        k, v = parse_override(item)
        overrides[k] = v
    if args.read_budget is not None:
        overrides["budget.read_tokens"] = args.read_budget
    base_cfg = SystemConfig.load(args.config, overrides)
    if args.ablation_group:
        table = yaml.safe_load(Path(args.ablations_file).read_text())
        rows = table["groups"][args.ablation_group]
        for row_name, row_overrides in rows.items():
            cfg = base_cfg.with_overrides(row_overrides).model_copy(update={"name": f"{base_cfg.name}/{args.ablation_group}={row_name}"})
            execute(cfg, args, {**overrides, **row_overrides}, questions)
        return 0
    execute(base_cfg, args, overrides, questions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
