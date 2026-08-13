from __future__ import annotations

import json
from decimal import Decimal

import pytest

from citedelta.answer.models import Answer, Citation, Refusal, RefusalReason
from citedelta.answer.service import AnswerService
from citedelta.fusion import FusedHit
from citedelta.retrieve import RetrievalTrace
from citedelta.temporal import AdmissibleSet
from substrate.llm import CompletionResponse, FakeCompletions, StopReason, TokenUsage


def _citation(chunk_id: int) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        citation_path=f"8 CFR 214.2(f)({chunk_id})",
        effective_from="2016-01-01",
        effective_to=None,
        text="Sample regulation text.",
        rrf_score=0.03,
        ranks={"lexical": 1, "vector": 1},
    )


def _trace(
    hits: list[FusedHit] | None = None, *, query: str = "What is the F-1 grace period?"
) -> RetrievalTrace:
    hits = hits if hits is not None else [FusedHit(1, 0.03, {"lexical": 1})]
    return RetrievalTrace(
        query=query,
        as_of="2026-08-11",
        selectivity=0.02,
        candidates_lexical=50,
        candidates_vector=50,
        fused=len(hits),
        hits=hits,
    )


def _admissible(ids: set[int]) -> AdmissibleSet:
    return AdmissibleSet(ids=frozenset(ids), label="2026-08-11", corpus_size=1000)


def _model_says(payload: dict[str, object]) -> CompletionResponse:
    return CompletionResponse(
        text=json.dumps(payload),
        usage=TokenUsage(100, 50),
        stop_reason=StopReason.END_TURN,
        model="claude-opus-5",
    )


def _service(fake: FakeCompletions) -> AnswerService:
    return AnswerService(fake, model="claude-opus-5")


@pytest.mark.asyncio
async def test_happy_path_returns_an_answer_with_citations() -> None:
    fake = FakeCompletions(
        responses=[
            _model_says(
                {
                    "out_of_scope": False,
                    "sufficient": True,
                    "answer": "Sixty days [1].",
                    "citations": [{"id": 1, "quote": "Sample regulation text."}],
                }
            )
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(),
        candidates=[_citation(1)],
        admissible=_admissible({1}),
    )
    assert isinstance(result, Answer)
    assert result.citations[0].chunk_id == 1


@pytest.mark.asyncio
async def test_fabricated_citation_destroys_the_answer() -> None:
    """The whole thesis, in one test. The model produced fluent, plausible
    prose; one cited id was never retrieved; nothing is shown."""
    fake = FakeCompletions(
        responses=[
            _model_says(
                {
                    "out_of_scope": False,
                    "sufficient": True,
                    "answer": "Sixty days [1], extendable to ninety [999].",
                    "citations": [
                        {"id": 1, "quote": "Sample regulation text."},
                        {"id": 999, "quote": "invented"},
                    ],
                }
            )
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(),
        candidates=[_citation(1)],
        admissible=_admissible({1}),
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.FABRICATED_CITATION
    assert not hasattr(result, "text")


@pytest.mark.asyncio
async def test_greeting_short_circuits_before_any_model_call() -> None:
    fake = FakeCompletions()  # no scripted response: a call would raise
    result = await _service(fake).answer(
        trace=_trace(query="hello"), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.GREETING
    assert fake.calls == []
    assert result.cost_usd == Decimal(0)


@pytest.mark.asyncio
async def test_empty_retrieval_refuses_before_spending_a_token() -> None:
    fake = FakeCompletions()  # no scripted response: a call would raise
    result = await _service(fake).answer(
        trace=_trace(hits=[]),
        candidates=[],
        admissible=_admissible(set()),
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.NO_ADMISSIBLE_SOURCE
    assert fake.calls == []
    assert result.cost_usd == Decimal(0)


@pytest.mark.asyncio
async def test_weak_retrieval_refuses_before_spending_a_token() -> None:
    fake = FakeCompletions()
    result = await _service(fake).answer(
        trace=_trace(hits=[FusedHit(1, 0.001, {"vector": 40})]),
        candidates=[_citation(1)],
        admissible=_admissible({1}),
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.LOW_CONFIDENCE
    assert fake.calls == []


@pytest.mark.asyncio
async def test_model_declaring_insufficiency_is_a_refusal_not_an_empty_answer() -> None:
    fake = FakeCompletions(
        responses=[
            _model_says({"out_of_scope": False, "sufficient": False, "answer": "", "citations": []})
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_provider_refusal_becomes_a_product_refusal() -> None:
    """HTTP 200 with an empty content list. Not an exception, not a crash."""
    fake = FakeCompletions(
        responses=[
            CompletionResponse(
                text="",
                usage=TokenUsage(100, 0),
                stop_reason=StopReason.REFUSAL,
                model="claude-opus-5",
                refusal_category="policy",
            )
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.PROVIDER_REFUSED


@pytest.mark.asyncio
async def test_unparseable_response_refuses_rather_than_crashing() -> None:
    fake = FakeCompletions(
        responses=[
            CompletionResponse(
                text="not json at all",
                usage=TokenUsage(10, 10),
                stop_reason=StopReason.END_TURN,
                model="claude-opus-5",
            )
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_only_admissible_text_reaches_the_model() -> None:
    """The temporal guarantee, asserted on the prompt itself: a chunk that
    was filtered out cannot be cited because it was never in the context."""
    fake = FakeCompletions(
        responses=[
            _model_says(
                {
                    "out_of_scope": False,
                    "sufficient": True,
                    "answer": "x [1].",
                    "citations": [{"id": 1, "quote": "Sample regulation text."}],
                }
            )
        ]
    )
    await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    prompt = fake.last.messages[0].content
    assert "[1]" in prompt
    assert "[999]" not in prompt
    assert "2026-08-11" in prompt


@pytest.mark.asyncio
async def test_out_of_scope_is_its_own_refusal() -> None:
    fake = FakeCompletions(
        responses=[
            _model_says({"out_of_scope": True, "sufficient": False, "answer": "", "citations": []})
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.OUT_OF_SCOPE


@pytest.mark.asyncio
async def test_scope_is_checked_before_sufficiency() -> None:
    """Both flags true-ish must resolve to OUT_OF_SCOPE — the two want
    different copy and collapsing them loses that."""
    fake = FakeCompletions(
        responses=[
            _model_says(
                {
                    "out_of_scope": True,
                    "sufficient": True,
                    "answer": "Paris [1].",
                    "citations": [{"id": 1, "quote": "x"}],
                }
            )
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.OUT_OF_SCOPE


@pytest.mark.asyncio
async def test_unverifiable_quote_discards_the_answer() -> None:
    """The most important test in the repo: an altered quote means the whole
    answer goes, exactly like a fabricated id."""
    fake = FakeCompletions(
        responses=[
            _model_says(
                {
                    "out_of_scope": False,
                    "sufficient": True,
                    "answer": "Sixty days [1].",
                    "citations": [{"id": 1, "quote": "words that are not in the source"}],
                }
            )
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert result.refused
    assert result.reason is RefusalReason.FABRICATED_CITATION
    assert not hasattr(result, "text")


@pytest.mark.asyncio
async def test_verified_quote_reaches_the_citation() -> None:
    """The sources panel bolds this string, so it has to survive the round trip."""
    fake = FakeCompletions(
        responses=[
            _model_says(
                {
                    "out_of_scope": False,
                    "sufficient": True,
                    "answer": "Sixty days [1].",
                    "citations": [{"id": 1, "quote": "Sample regulation text."}],
                }
            )
        ]
    )
    result = await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert isinstance(result, Answer)
    assert result.citations[0].quote == "Sample regulation text."
