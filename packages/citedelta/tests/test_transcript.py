from __future__ import annotations

from datetime import date
from decimal import Decimal

from citedelta.answer.models import Answer, Refusal, RefusalReason
from citedelta.retrieve import RetrievalTrace
from citedelta.web.transcript import (
    Rupture,
    TurnView,
    amendments_between,
    build_transcript,
)


def _answer() -> Answer:
    trace = RetrievalTrace(query="q", as_of="2026-08-13", selectivity=0.02, fused=1, hits=[])
    return Answer(
        query="q",
        as_of="2026-08-13",
        text="answer",
        citations=(),
        trace=trace,
        cost_usd=Decimal(0),
        latency_ms=1.0,
    )


def _refusal() -> Refusal:
    return Refusal(
        query="q",
        as_of="2026-08-13",
        reason=RefusalReason.OUT_OF_SCOPE,
        detail="outside",
        trace=None,
        cost_usd=Decimal(0),
        latency_ms=1.0,
    )


DATES = [
    date(2016, 12, 23),
    date(2019, 3, 12),
    date(2020, 5, 1),
    date(2024, 4, 12),
    date(2026, 7, 17),
]


def _turn(as_of: date, *, continuation: bool = False) -> TurnView:
    return TurnView(
        question="q", as_of=as_of, result=_answer(), resolved_query=None, continuation=continuation
    )


def test_same_date_produces_no_rupture() -> None:
    out = build_transcript(
        [_turn(date(2026, 8, 13)), _turn(date(2026, 8, 13))], amendment_dates=DATES
    )
    assert not any(isinstance(e, Rupture) for e in out)


def test_changed_date_inserts_a_rupture_between_them() -> None:
    out = build_transcript(
        [_turn(date(2026, 8, 13)), _turn(date(2019, 4, 1))], amendment_dates=DATES
    )
    rupture = out[1]
    assert isinstance(rupture, Rupture)
    assert rupture.earlier


def test_forward_travel_is_marked_as_later() -> None:
    out = build_transcript(
        [_turn(date(2019, 4, 1)), _turn(date(2026, 8, 13))], amendment_dates=DATES
    )
    rupture = out[1]
    assert isinstance(rupture, Rupture)
    assert not rupture.earlier
    assert "later" in rupture.count_label


def test_amendments_are_counted_strictly_between() -> None:
    """A boundary amendment is the law you are now reading, not one you
    skipped — counting it would overstate the distance at most real dates."""
    assert amendments_between(date(2019, 3, 12), date(2026, 7, 17), DATES) == 2
    assert amendments_between(date(2019, 3, 11), date(2026, 7, 18), DATES) == 4


def test_crossing_no_amendments_says_so() -> None:
    """Same window, so the answer will be identical — the UI has to explain
    that rather than let it look like a bug."""
    r = Rupture(as_of=date(2022, 4, 1), amendments_between=0, earlier=False)
    assert "no amendments" in r.count_label


def test_one_amendment_is_singular() -> None:
    assert (
        "1 amendment "
        in Rupture(as_of=date(2020, 6, 1), amendments_between=1, earlier=True).count_label
    )


def test_stamp_splits_day_from_year() -> None:
    t = _turn(date(2019, 4, 12))
    assert t.stamp_day == "12 Apr"
    assert t.stamp_year == "2019"


def test_refusals_render_through_the_same_pipeline() -> None:
    turn = TurnView(
        question="q",
        as_of=date(2026, 8, 13),
        result=_refusal(),
        resolved_query=None,
        continuation=False,
    )
    out = build_transcript([turn], amendment_dates=DATES)
    rendered = out[0]
    assert isinstance(rendered, TurnView)
    assert rendered.result.refused
