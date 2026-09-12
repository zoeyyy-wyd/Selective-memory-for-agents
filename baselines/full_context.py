"""Full context: the entire history goes to the answering model. Dev questions only (cost)."""

from __future__ import annotations

from baselines import BaselineOutput, answer_from_context, render_turns
from evals.longmemeval.data import LMEQuestion
from smem.llm import LLM
from smem.tokens import count_tokens


class FullContextBaseline:
    name = "full_context"

    def __init__(self, llm: LLM | None):
        self.llm = llm

    def answer_question(self, q: LMEQuestion) -> BaselineOutput:
        context = render_turns(q)
        answer = answer_from_context(self.llm, q.question, context, q.question_date,
                                     max_tokens=getattr(self, "answer_max_tokens", 300))
        return BaselineOutput(answer, count_tokens(context), len(context.splitlines()), context)
