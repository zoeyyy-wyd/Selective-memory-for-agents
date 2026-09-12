"""Wrapper around the official LongMemEval judge (evaluate_qa.py). The prompts below are a port of
the upstream templates; the label is "yes" in the judge's response, as upstream. Dev runs use
gpt-4.1-mini, the final test run uses GPT-4o at temperature 0. `ExactMatchJudge` is the offline
stand-in for tests and dry runs."""

from __future__ import annotations

import re
from typing import Protocol

from evals.longmemeval.data import LMEQuestion
from smem.llm import LLM

_GENERIC = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
            "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
            "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
            "the response only contains a subset of the information required by the answer, answer no. \n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
_TEMPORAL = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
             "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
             "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
             "the response only contains a subset of the information required by the answer, answer no. In addition, "
             "do not penalize off-by-one errors for the number of days. If the question asks for the number of "
             "days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
             "is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
             "{}\n\nIs the model response correct? Answer yes or no only.")
_UPDATE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
           "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
           "information along with an updated answer, the response should be considered as correct as long as the "
           "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
           "Is the model response correct? Answer yes or no only.")
_PREFERENCE = ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
               "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does "
               "not need to reflect all the points in the rubric. The response is correct as long as it recalls and "
               "utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: "
               "{}\n\nIs the model response correct? Answer yes or no only.")
_ABSTENTION = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer "
               "yes if the model correctly identifies the question as unanswerable. The model could say that the "
               "information is incomplete, or some other information is given but the asked information is not.\n\n"
               "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question "
               "as unanswerable? Answer yes or no only.")


def judge_prompt(q: LMEQuestion, response: str) -> str:
    if q.is_abstention:
        template = _ABSTENTION
    elif q.question_type == "temporal-reasoning":
        template = _TEMPORAL
    elif q.question_type == "knowledge-update":
        template = _UPDATE
    elif q.question_type == "single-session-preference":
        template = _PREFERENCE
    else:
        template = _GENERIC
    return template.format(q.question, q.answer, response)


class Judge(Protocol):
    name: str

    def judge(self, q: LMEQuestion, response: str) -> bool | None: ...


_VERDICT_RE = re.compile(r"\b(yes|no)\b")


def parse_verdict(text: str) -> bool | None:
    """First standalone yes/no wins; None when the judge said neither.

    Upstream tests `"yes" in response.lower()`, which agrees with this on the terse output the prompt
    asks for but calls "no, not yes" a yes. None is not False: `summarise` drops unjudged records
    from accuracy, so a judge that fails to answer shows up as a smaller n instead of silently
    scoring every response wrong."""
    m = _VERDICT_RE.search(text.strip().lower())
    return None if m is None else m.group(1) == "yes"


class LLMJudge:
    """`max_tokens` defaults to 512, not the 10 the terse prompt needs. A reasoning model spends its
    budget before emitting any text -- openai/gpt-5-mini returns an empty string at 10 -- and an
    empty verdict used to read as "answered wrong" with nothing logged. Output here is a word or two,
    so the extra headroom costs nothing on a non-reasoning judge."""

    def __init__(self, llm: LLM, max_tokens: int = 512):
        self.llm = llm
        self.max_tokens = max_tokens
        self.name = f"llm:{llm.model}"
        self.n_empty = 0
        self.n_unparsed = 0

    def judge(self, q: LMEQuestion, response: str) -> bool | None:
        out = self.llm.complete("", judge_prompt(q, response), temperature=0.0, max_tokens=self.max_tokens)
        verdict = parse_verdict(out)
        if not out.strip():
            self.n_empty += 1
        elif verdict is None:
            self.n_unparsed += 1
        return verdict

    def health(self) -> str | None:
        """Non-None means the judge itself misbehaved and the accuracy column is not trustworthy."""
        bad = self.n_empty + self.n_unparsed
        if not bad:
            return None
        return (f"judge {self.llm.model} returned no verdict on {bad} calls "
                f"({self.n_empty} empty, {self.n_unparsed} unparseable); those questions are unjudged. "
                f"An empty verdict usually means a reasoning model ran out of max_tokens={self.max_tokens}.")


class ExactMatchJudge:
    """Offline stand-in: abstention questions are correct when the model abstains; otherwise the gold
    answer string must appear in the response."""

    name = "exact"

    def judge(self, q: LMEQuestion, response: str) -> bool | None:
        r = response.strip().lower()
        if q.is_abstention:
            return r.startswith("i don't know") or "not" in r and "information" in r
        gold = q.answer.strip().lower()
        return bool(gold) and gold in r


class NoJudge:
    name = "none"

    def judge(self, q: LMEQuestion, response: str) -> bool:  # pragma: no cover - trivial
        raise RuntimeError("no judge configured")
