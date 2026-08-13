from __future__ import annotations

import json
from datetime import date

import pytest

from citedelta.answer.resolve import RESOLVER_MODEL, _safe_as_of, resolve_followup
from citedelta.api.traces import PriorTurn
from substrate.llm import CompletionResponse, FakeCompletions, StopReason, TokenUsage

TODAY = date(2026, 8, 13)
SINCE = date(2016, 12, 23)


def _says(**payload: object) -> CompletionResponse:
    return CompletionResponse(
        text=json.dumps(payload),
        usage=TokenUsage(300, 60),
        stop_reason=StopReason.END_TURN,
        model=RESOLVER_MODEL,
    )


def _history() -> list[PriorTurn]:
    return [
        PriorTurn(
            query="What is the grace period after F-1 program completion?",
            resolved_query="What is the grace period after F-1 program completion?",
            as_of=TODAY,
            answered=True,
        )
    ]


@pytest.mark.asyncio
async def test_first_turn_spends_nothing() -> None:
    """Most turns are first turns. They must not pay for a resolver call."""
    fake = FakeCompletions()  # a call would raise
    r = await resolve_followup(
        fake,
        question="What is the grace period?",
        history=[],
        current_as_of=TODAY,
        corpus_since=SINCE,
        today=TODAY,
    )
    assert fake.calls == []
    assert r.as_of is None
    assert not r.is_followup


@pytest.mark.asyncio
async def test_followup_gains_a_subject_and_a_date() -> None:
    fake = FakeCompletions(
        responses=[
            _says(
                is_followup=True,
                standalone_question="What was the grace period after F-1 program completion?",
                as_of="2019-04-12",
            )
        ]
    )
    r = await resolve_followup(
        fake,
        question="What about in 2019?",
        history=_history(),
        current_as_of=TODAY,
        corpus_since=SINCE,
        today=TODAY,
    )
    assert r.is_followup
    assert "grace period" in r.standalone_question
    assert r.as_of == date(2019, 4, 12)


@pytest.mark.asyncio
async def test_uses_the_cheap_model() -> None:
    fake = FakeCompletions(responses=[_says(is_followup=True, standalone_question="x", as_of="")])
    await resolve_followup(
        fake,
        question="what about then?",
        history=_history(),
        current_as_of=TODAY,
        corpus_since=SINCE,
        today=TODAY,
    )
    assert fake.last.model == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_new_topic_mid_conversation_is_not_a_followup() -> None:
    fake = FakeCompletions(
        responses=[
            _says(
                is_followup=False,
                standalone_question="Can an F-1 student transfer schools?",
                as_of="",
            )
        ]
    )
    r = await resolve_followup(
        fake,
        question="Can an F-1 student transfer schools?",
        history=_history(),
        current_as_of=TODAY,
        corpus_since=SINCE,
        today=TODAY,
    )
    assert not r.is_followup
    assert r.as_of is None


@pytest.mark.asyncio
async def test_malformed_resolution_degrades_to_the_raw_question() -> None:
    """A cheap auxiliary call must not be able to take down the turn."""
    fake = FakeCompletions(
        responses=[
            CompletionResponse(
                text="not json",
                usage=TokenUsage(10, 10),
                stop_reason=StopReason.END_TURN,
                model=RESOLVER_MODEL,
            )
        ]
    )
    r = await resolve_followup(
        fake,
        question="what about then?",
        history=_history(),
        current_as_of=TODAY,
        corpus_since=SINCE,
        today=TODAY,
    )
    assert r.standalone_question == "what about then?"
    assert r.as_of is None


@pytest.mark.asyncio
async def test_provider_refusal_degrades_the_same_way() -> None:
    fake = FakeCompletions(
        responses=[
            CompletionResponse(
                text="",
                usage=TokenUsage(10, 0),
                stop_reason=StopReason.REFUSAL,
                model=RESOLVER_MODEL,
            )
        ]
    )
    r = await resolve_followup(
        fake,
        question="what about then?",
        history=_history(),
        current_as_of=TODAY,
        corpus_since=SINCE,
        today=TODAY,
    )
    assert r.standalone_question == "what about then?"


def test_dates_are_clamped_into_the_corpus_window() -> None:
    """A model answering 'back in 2010' would otherwise produce an empty
    admissible set and a refusal that looks like a bug."""
    assert _safe_as_of("2010-01-01", corpus_since=SINCE, today=TODAY) == SINCE
    assert _safe_as_of("2099-01-01", corpus_since=SINCE, today=TODAY) == TODAY
    assert _safe_as_of("2019-04-12", corpus_since=SINCE, today=TODAY) == date(2019, 4, 12)


def test_unparseable_and_empty_dates_mean_unchanged() -> None:
    assert _safe_as_of("", corpus_since=SINCE, today=TODAY) is None
    assert _safe_as_of("sometime in 2019", corpus_since=SINCE, today=TODAY) is None
    assert _safe_as_of("2019-13-45", corpus_since=SINCE, today=TODAY) is None
