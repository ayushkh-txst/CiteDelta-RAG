from __future__ import annotations

from datetime import date
from decimal import Decimal

from citedelta.answer.models import Answer, Citation
from citedelta.eval.cases import CaseClass, EvalCase
from citedelta.eval.graders import _citations_hold, grade
from citedelta.retrieve import RetrievalTrace


def _trace() -> RetrievalTrace:
    return RetrievalTrace(query="q", as_of="2026-08-13", selectivity=0.02, fused=1, hits=[])


def _case() -> EvalCase:
    return EvalCase(
        id="g-01",
        cls=CaseClass.FACTUAL,
        query="q",
        as_of=date(2026, 8, 13),
        expects_refusal=False,
        expected_reason=None,
        expected_citations=("8 CFR 214.2(f)(5)(iv)",),
        must_not_cite=(),
        verified_by="tester",
    )


def test_citations_valid_can_actually_fail() -> None:
    """The regression guard for the vacuous grader: it used to re-derive its
    answer from the code it was auditing, so it reported 1.00 even with the
    guarantee deleted."""
    bad = Answer(
        query="q",
        as_of="2026-08-13",
        text="x [1]",
        citations=(
            Citation(
                chunk_id=1,
                citation_path="8 CFR 214.2(f)(5)(iv)",
                effective_from="2016-01-01",
                effective_to=None,
                text="the real source text",
                rrf_score=0.03,
                quote="words not present",
            ),
        ),
        trace=_trace(),
        cost_usd=Decimal(0),
        latency_ms=1.0,
    )
    assert not _citations_hold(_case(), bad, [bad.citations[0]])


def test_grade_rejects_an_answer_whose_quote_is_invented() -> None:
    """The grade() metric must agree with _citations_hold, since that is the
    guarantee the eval exists to check."""
    bad = Answer(
        query="q",
        as_of="2026-08-13",
        text="x [1]",
        citations=(
            Citation(
                chunk_id=1,
                citation_path="8 CFR 214.2(f)(5)(iv)",
                effective_from="2016-01-01",
                effective_to=None,
                text="the real source text",
                rrf_score=0.03,
                quote="made up",
            ),
        ),
        trace=_trace(),
        cost_usd=Decimal(0),
        latency_ms=1.0,
    )
    score = grade(_case(), bad, [bad.citations[0]])
    assert not score.citations_valid
    assert not score.passed
