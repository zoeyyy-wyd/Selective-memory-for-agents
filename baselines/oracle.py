"""Oracle: inject only the annotated evidence sessions. The answering model's ceiling."""

from __future__ import annotations

from baselines import BaselineOutput, answer_from_context, render_turns
from evals.longmemeval.data import LMEQuestion
from smem.llm import LLM
from smem.tokens import count_tokens


class OracleBaseline:
    name = "oracle"

    def __init__(self, llm: LLM | None, whole_sessions: bool = True):
        self.llm = llm
        self.whole_sessions = whole_sessions

    def answer_question(self, q: LMEQuestion) -> BaselineOutput:
        context = render_turns(q, q.evidence_session_ids, only_evidence_turns=not self.whole_sessions)
        answer = answer_from_context(self.llm, q.question, context, q.question_date)
        return BaselineOutput(answer, count_tokens(context), len(context.splitlines()), context,
                              {"evidence_sessions": sorted(q.evidence_session_ids)})
