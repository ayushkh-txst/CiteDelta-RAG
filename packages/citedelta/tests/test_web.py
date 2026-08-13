from __future__ import annotations

from datetime import date
from decimal import Decimal

from citedelta.answer.models import Answer, Citation, Refusal, RefusalReason
from citedelta.api.app import templates
from citedelta.retrieve import RetrievalTrace
from citedelta.web.filters import citation_chips
from citedelta.web.transcript import TurnView


def _citation(chunk_id: int) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        citation_path=f"8 CFR 214.2(f)({chunk_id})",
        effective_from="2016-01-01",
        effective_to=None,
        text="text",
        rrf_score=0.5,
    )


def test_chips_escape_model_text_first() -> None:
    text = "He wrote <script>alert(1)</script> then cited [7]."
    out = str(citation_chips(text, [_citation(7)]))

    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert 'class="chip"' in out


def test_chips_map_chunk_id_to_display_ordinal() -> None:
    cits = [_citation(100), _citation(200), _citation(300)]
    out = str(citation_chips("see [200] and [300]", cits))

    assert '<a class="chip" href="#cite-2">2</a>' in out
    assert '<a class="chip" href="#cite-3">3</a>' in out
    assert "[200]" not in out


def test_chips_leave_unmapped_ids_plain() -> None:
    out = str(citation_chips("no such [9999]", [_citation(7)]))

    assert "[9999]" in out
    assert 'class="chip"' not in out


def _refusal(reason: str) -> Refusal:
    return Refusal(
        query="q",
        as_of="2026-08-13",
        reason=RefusalReason(reason),
        detail="something happened",
        trace=None,
        cost_usd=Decimal(0),
        latency_ms=1.0,
    )


def _greeting_refusal() -> Refusal:
    return _refusal("greeting")


def _answer() -> Answer:
    trace = RetrievalTrace(query="q", as_of="2026-08-13", selectivity=0.02, fused=1, hits=[])
    return Answer(
        query="q",
        as_of="2026-08-13",
        text="answer [1]",
        citations=(_citation(1),),
        trace=trace,
        cost_usd=Decimal(0),
        latency_ms=1.0,
    )


def _render_turn(
    result: Answer | Refusal, *, question: str = "q", resolved_query: str | None = None
) -> str:
    turn = TurnView(
        question=question,
        as_of=date(2026, 8, 13),
        result=result,
        resolved_query=resolved_query,
        continuation=False,
    )
    return templates.env.get_template("partials/turn.html").render(
        turn=turn,
        candidates=[],
        cited_ids=[],
        max_score=1.0,
        trace_id=1,
        compare_options=[],
    )


def test_greeting_renders_conversationally_not_as_a_refusal() -> None:
    html = _render_turn(_greeting_refusal())
    assert "greeting" in html
    assert "declined" not in html
    assert "No search run" in html


def test_alert_styling_only_for_a_discarded_answer() -> None:
    assert "declined hard" in _render_turn(_refusal("fabricated_citation"))
    for reason in ("out_of_scope", "low_confidence", "insufficient_evidence"):
        assert "declined hard" not in _render_turn(_refusal(reason))


def test_resolved_query_shown_only_when_it_differs() -> None:
    assert "searched:" in _render_turn(
        _answer(), question="what about then?", resolved_query="grace period after F-1"
    )
    assert "searched:" not in _render_turn(
        _answer(), question="grace period after F-1", resolved_query=None
    )
