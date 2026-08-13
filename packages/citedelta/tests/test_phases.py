from __future__ import annotations

import json

import pytest

from citedelta.answer.models import Citation
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


def _trace(hits: list[FusedHit] | None = None) -> RetrievalTrace:
    hits = hits if hits is not None else [FusedHit(1, 0.03, {"lexical": 1})]
    return RetrievalTrace(
        query="What is the F-1 grace period?",
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
async def test_phases_are_reported_in_order() -> None:
    seen: list[str] = []

    async def hook(text: str) -> None:
        seen.append(text)

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
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1}), on_phase=hook
    )

    assert any("Reading" in p for p in seen)
    assert any("Drafting" in p for p in seen)
    assert any("Verifying" in p for p in seen)
    assert seen.index(next(p for p in seen if "Drafting" in p)) < seen.index(
        next(p for p in seen if "Verifying" in p)
    )


@pytest.mark.asyncio
async def test_no_hook_changes_nothing() -> None:
    """The JSON API and the eval must run exactly as before."""
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
    result = await _service(fake).answer(
        trace=_trace(), candidates=[_citation(1)], admissible=_admissible({1})
    )
    assert not result.refused


@pytest.mark.asyncio
async def test_a_refusal_still_reports_no_phantom_phases() -> None:
    """A pre-flight refusal spends no tokens, so it must not claim to have
    drafted anything."""
    seen: list[str] = []

    async def hook(text: str) -> None:
        seen.append(text)

    fake = FakeCompletions()
    await _service(fake).answer(
        trace=_trace(hits=[]), candidates=[], admissible=_admissible(set()), on_phase=hook
    )
    assert not any("Drafting" in p for p in seen)
