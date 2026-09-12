"""Evaluation metrics (plan section 10). Beyond accuracy, three evidence metrics decompose a wrong
answer into never written / evicted / not read / read but answered wrong:

  evidence write rate      |E ∩ written|  / |E|
  evidence survival rate   |E ∩ store|    / |E|   (lenient: an evicted episode whose accepted summary is stored counts)
  evidence injection rate  |E ∩ injected| / |E|

E = candidates extracted from the annotated evidence sessions (episodes restricted to has_answer turns
when the label exists). Every aggregate carries a bootstrap 95% CI resampled by question."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from evals.longmemeval.data import LMEQuestion
from smem.system import SelectiveMemory


@dataclass
class QuestionRecord:
    question_id: str
    question_type: str
    is_abstention: bool
    question: str
    gold: str
    answer: str
    abstained: bool
    correct: bool | None
    n_evidence: int = 0
    n_written: int = 0
    n_survived: int = 0
    n_survived_strict: int = 0
    n_injected: int = 0
    injected_tokens: int = 0
    injected_entries: int = 0
    store_tokens: int = 0
    store_entries: int = 0
    history_tokens: int = 0
    invalid_citations: int = 0
    config_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def stratum(self) -> str:
        return f"{self.question_type}{'_abs' if self.is_abstention else ''}"


def evidence_counts(mem: SelectiveMemory, q: LMEQuestion, injected_ids: set[str]) -> dict[str, int]:
    turn_filter = q.evidence_turns or None
    evidence = mem.candidates_from_sessions(q.evidence_session_ids, turn_filter)
    written = evidence & mem.writer.written_ids
    store = mem.surviving_ids()
    # lenient: a member of an accepted, stored summary counts as surviving / injected through it
    covered_by_summary = set()
    injected_by_summary = set()
    for summary_id, members in mem.summary_sources().items():
        if summary_id in store:
            covered_by_summary |= members
        if summary_id in injected_ids:
            injected_by_summary |= members
    survived_strict = evidence & store
    survived = survived_strict | (evidence & covered_by_summary)
    injected = (evidence & injected_ids) | (evidence & injected_by_summary)
    return {
        "n_evidence": len(evidence), "n_written": len(written), "n_survived": len(survived),
        "n_survived_strict": len(survived_strict), "n_injected": len(injected),
    }


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> dict[str, float | int]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": mean, "lo": mean, "hi": mean, "n": 1}
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"mean": mean, "lo": means[int(0.025 * n_boot)], "hi": means[min(int(0.975 * n_boot), n_boot - 1)], "n": len(vals)}


def aggregate(records: list[QuestionRecord], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    judged = [r for r in records if r.correct is not None]
    answerable = [r for r in records if not r.is_abstention]
    unanswerable = [r for r in records if r.is_abstention]
    by_type: dict[str, list[QuestionRecord]] = defaultdict(list)
    for r in judged:
        by_type[r.stratum].append(r)
    out: dict[str, Any] = {
        "n": len(records),
        "n_judged": len(judged),   # < n means the judge returned no verdict on some questions
        "accuracy": bootstrap_ci([float(r.correct) for r in judged], n_boot, seed),
        "accuracy_by_type": {t: bootstrap_ci([float(r.correct) for r in rs], n_boot, seed) for t, rs in sorted(by_type.items())},
        "evidence_write_rate": bootstrap_ci([_rate(r.n_written, r.n_evidence) for r in answerable], n_boot, seed),
        "evidence_survival_rate": bootstrap_ci([_rate(r.n_survived, r.n_evidence) for r in answerable], n_boot, seed),
        "evidence_survival_rate_strict": bootstrap_ci([_rate(r.n_survived_strict, r.n_evidence) for r in answerable], n_boot, seed),
        "evidence_injection_rate": bootstrap_ci([_rate(r.n_injected, r.n_evidence) for r in answerable], n_boot, seed),
        "correct_abstention_rate": bootstrap_ci([float(r.abstained) for r in unanswerable], n_boot, seed),
        "false_abstention_rate": bootstrap_ci([float(r.abstained) for r in answerable], n_boot, seed),
        "tokens_per_query": bootstrap_ci([float(r.injected_tokens) for r in records], n_boot, seed),
        "entries_per_query": bootstrap_ci([float(r.injected_entries) for r in records], n_boot, seed),
        "store_tokens": bootstrap_ci([float(r.store_tokens) for r in records], n_boot, seed),
        "store_fraction": bootstrap_ci([r.store_tokens / r.history_tokens for r in records if r.history_tokens], n_boot, seed),
        "invalid_citation_rate": bootstrap_ci([float(r.invalid_citations > 0) for r in records], n_boot, seed),
    }
    return out


def format_summary(summary: dict[str, Any]) -> str:
    def cell(ci: dict) -> str:
        if ci is None or ci.get("mean") is None:
            return "   n/a"
        return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] (n={ci['n']})"

    lines = [f"n = {summary['n']}"]
    for key in ("accuracy", "evidence_write_rate", "evidence_survival_rate", "evidence_survival_rate_strict",
                "evidence_injection_rate", "correct_abstention_rate", "false_abstention_rate", "tokens_per_query",
                "entries_per_query", "store_tokens", "store_fraction", "invalid_citation_rate"):
        lines.append(f"{key:32s} {cell(summary[key])}")
    if summary["accuracy_by_type"]:
        lines.append("accuracy by type:")
        for t, ci in summary["accuracy_by_type"].items():
            lines.append(f"  {t:32s} {cell(ci)}")
    return "\n".join(lines)
