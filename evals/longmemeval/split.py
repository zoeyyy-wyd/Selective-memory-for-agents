"""Stratified dev/test split. 100 dev questions for tuning, 400 test questions touched only at the
end. The split is a pure function of (question ids, seed) and is also written to disk for the record."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from evals.longmemeval.data import LMEQuestion


def stratified_split(questions: list[LMEQuestion], n_dev: int = 100, seed: int = 0
                     ) -> tuple[list[LMEQuestion], list[LMEQuestion]]:
    by_stratum: dict[str, list[LMEQuestion]] = defaultdict(list)
    for q in questions:
        by_stratum[q.stratum].append(q)
    rng = random.Random(seed)
    total = len(questions)
    dev_ids: set[str] = set()
    # largest-remainder allocation so strata sizes sum exactly to n_dev
    quotas = {k: n_dev * len(v) / total for k, v in by_stratum.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remainder = n_dev - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: -(quotas[k] - alloc[k]))[:remainder]:
        alloc[k] += 1
    for k, items in sorted(by_stratum.items()):
        ids = sorted(q.question_id for q in items)
        rng.shuffle(ids)
        dev_ids.update(ids[: alloc[k]])
    dev = [q for q in questions if q.question_id in dev_ids]
    test = [q for q in questions if q.question_id not in dev_ids]
    return dev, test


def save_split(dev: list[LMEQuestion], test: list[LMEQuestion], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"dev": [q.question_id for q in dev], "test": [q.question_id for q in test]}, indent=1))


def load_split(questions: list[LMEQuestion], path: str | Path) -> tuple[list[LMEQuestion], list[LMEQuestion]]:
    ids = json.loads(Path(path).read_text())
    dev_ids, test_ids = set(ids["dev"]), set(ids["test"])
    return [q for q in questions if q.question_id in dev_ids], [q for q in questions if q.question_id in test_ids]
