import json
from datetime import datetime

from evals.longmemeval.data import load_longmemeval
from evals.longmemeval.judge import ExactMatchJudge, judge_prompt
from evals.longmemeval.split import stratified_split
from evals.metrics import QuestionRecord, aggregate, bootstrap_ci
from evals.run import main


def make_item(qid, qtype, question, answer, evidence_text, n_filler=3):
    sessions, ids, dates = [], [], []
    for i in range(n_filler):
        sessions.append([{"role": "user", "content": f"I watched a documentary about volcano number {i} yesterday."},
                         {"role": "assistant", "content": "Sounds interesting."}])
        ids.append(f"filler_{qid}_{i}")
        dates.append(f"2023/01/{10 + i:02d} (Tue) 10:00")
    sessions.insert(1, [{"role": "user", "content": evidence_text, "has_answer": True},
                        {"role": "assistant", "content": "Noted."}])
    ids.insert(1, f"answer_{qid}")
    dates.insert(1, "2023/01/11 (Wed) 12:00")
    return {"question_id": qid, "question_type": qtype, "question": question, "answer": answer,
            "question_date": "2023/03/01 (Wed) 09:00", "answer_session_ids": [f"answer_{qid}"],
            "haystack_dates": dates, "haystack_session_ids": ids, "haystack_sessions": sessions}


def write_dataset(path):
    items = [
        make_item("q1", "single-session-user", "Which city does the user live in now?", "Boston", "I just moved to Boston, renting near Kendall."),
        make_item("q2", "knowledge-update", "What is the user allergic to?", "peanuts", "I am allergic to peanuts so no satay please."),
        make_item("q3_abs", "single-session-user", "What is the user's car model?", "unanswerable", "I adopted a cat named Milo."),
        make_item("q4", "multi-session", "What is the user's favorite cuisine?", "Thai", "My favorite cuisine is Thai."),
    ]
    path.write_text(json.dumps(items))


def test_loader_split_and_offline_run(tmp_path, capsys):
    data = tmp_path / "mini.json"
    write_dataset(data)
    qs = load_longmemeval(data)
    assert len(qs) == 4 and qs[0].evidence_turns == {"answer_q1": {0}} and qs[2].is_abstention
    assert qs[0].question_date == datetime(2023, 3, 1, 9, 0)
    dev, test = stratified_split(qs, n_dev=2, seed=0)
    assert len(dev) == 2 and len(test) == 2 and {q.question_id for q in dev}.isdisjoint({q.question_id for q in test})
    out = tmp_path / "results"
    rc = main(["--split", "all", "--data", str(data), "--config", "configs/offline.yaml", "--backend", "offline",
               "--judge", "exact", "--store-budget", "0.5", "--read-budget", "300", "--out", str(out),
               "--n-boot", "50", "--quiet", "--dev-size", "2"])
    assert rc == 0
    run_dirs = list(out.iterdir())
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text())
    records = [json.loads(l) for l in (run_dirs[0] / "records.jsonl").read_text().splitlines()]
    assert summary["n"] == 4 and len(records) == 4
    assert summary["evidence_write_rate"]["mean"] > 0
    assert all(r["n_evidence"] >= 1 for r in records)
    assert all(r["store_tokens"] <= r["extra"]["store_budget"] for r in records)
    # rerunning reuses the records instead of recomputing
    rc = main(["--split", "all", "--data", str(data), "--config", "configs/offline.yaml", "--backend", "offline",
               "--judge", "exact", "--store-budget", "0.5", "--read-budget", "300", "--out", str(out),
               "--n-boot", "50", "--quiet", "--dev-size", "2"])
    assert rc == 0 and len(list(out.iterdir())) == 1


def test_baselines_offline(tmp_path):
    data = tmp_path / "mini.json"
    write_dataset(data)
    from baselines import build_baseline
    from smem.config import SystemConfig

    cfg = SystemConfig.load("configs/offline.yaml", {"budget.read_tokens": 150})
    q = load_longmemeval(data)[0]
    for name in ("oracle", "full_context", "naive_rag"):
        out = build_baseline(name, cfg, "offline").answer_question(q)
        assert out.answer and out.injected_tokens > 0
    oracle = build_baseline("oracle", cfg, "offline").answer_question(q)
    assert "Boston" in oracle.context and "volcano" not in oracle.context
    rag = build_baseline("naive_rag", cfg, "offline").answer_question(q)
    assert rag.injected_tokens <= 150


def test_metrics_and_ci():
    recs = [QuestionRecord(f"q{i}", "multi-session", False, "q", "a", "a", False, i % 2 == 0, n_evidence=4, n_written=4,
                           n_survived=2, n_survived_strict=2, n_injected=1, injected_tokens=100, history_tokens=1000,
                           store_tokens=250) for i in range(10)]
    recs.append(QuestionRecord("abs", "multi-session", True, "q", "a", "I don't know.", True, True, injected_tokens=50,
                               history_tokens=1000, store_tokens=250))
    s = aggregate(recs, n_boot=200)
    assert s["accuracy"]["mean"] == 6 / 11
    assert s["evidence_write_rate"]["mean"] == 1.0 and s["evidence_survival_rate"]["mean"] == 0.5
    assert s["evidence_injection_rate"]["mean"] == 0.25 and s["correct_abstention_rate"]["mean"] == 1.0
    assert s["store_fraction"]["mean"] == 0.25
    ci = bootstrap_ci([0.0, 1.0] * 20, n_boot=300)
    assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["lo"] < ci["hi"]
    assert bootstrap_ci([])["mean"] is None


def test_judge_prompts_and_exact_judge(tmp_path):
    data = tmp_path / "mini.json"
    write_dataset(data)
    qs = load_longmemeval(data)
    assert "off-by-one" not in judge_prompt(qs[0], "x") and "unanswerable" in judge_prompt(qs[2], "x")
    j = ExactMatchJudge()
    assert j.judge(qs[0], "The user lives in Boston [f_1].") and not j.judge(qs[0], "Seattle")
    assert j.judge(qs[2], "I don't know.") and not j.judge(qs[2], "A Toyota")


def test_empty_judge_verdict_is_ungraded_not_wrong():
    """A reasoning judge out of max_tokens returns "". That used to read as 'answered wrong' with
    nothing logged, i.e. a silent accuracy of 0."""
    from evals.longmemeval.judge import LLMJudge, parse_verdict
    from smem.llm import ScriptedLLM

    assert parse_verdict("") is None
    assert parse_verdict("no, the answer is not yes") is False   # upstream substring rule says True
    assert parse_verdict("Yes.") is True

    j = LLMJudge(ScriptedLLM(["", "yes", "I cannot tell"]))
    assert j.max_tokens == 512   # not the 10 the terse prompt needs: a reasoning judge needs room
    assert j.health() is None    # nothing has gone wrong yet
    j.n_empty, j.n_unparsed = 3, 1
    assert "4 calls" in j.health() and "3 empty" in j.health()


def test_aggregate_excludes_unjudged_from_accuracy():
    from evals.metrics import QuestionRecord, aggregate

    def rec(qid, correct):
        return QuestionRecord(question_id=qid, question_type="multi-session", is_abstention=False,
                              question="q", gold="g", answer="a", abstained=False, correct=correct)

    s = aggregate([rec("a", True), rec("b", False), rec("c", None)], n_boot=50, seed=0)
    assert s["n"] == 3 and s["n_judged"] == 2
    assert s["accuracy"]["mean"] == 0.5      # the unjudged one is dropped, not counted wrong
