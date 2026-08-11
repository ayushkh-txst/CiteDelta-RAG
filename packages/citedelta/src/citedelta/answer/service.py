"""Retrieve, gate, generate, validate. The whole answer path in one place."""

from __future__ import annotations

import json
import time
from decimal import Decimal

import structlog

from citedelta.answer.gate import pre_flight
from citedelta.answer.models import (
    Answer,
    AnswerResult,
    Citation,
    Refusal,
    RefusalReason,
)
from citedelta.answer.prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_message
from citedelta.answer.rerank import PassthroughReranker, Reranker
from citedelta.answer.validator import validate_citations
from citedelta.retrieve import RetrievalTrace
from citedelta.temporal import AdmissibleSet
from substrate.llm import (
    CompletionError,
    CompletionRequest,
    Completions,
    Message,
    Role,
)

log = structlog.get_logger(__name__)


class AnswerService:
    def __init__(
        self,
        llm: Completions,
        *,
        model: str,
        max_tokens: int = 2048,
        reranker: Reranker | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._max_tokens = max_tokens
        self._reranker = reranker or PassthroughReranker()

    async def answer(
        self,
        *,
        trace: RetrievalTrace,
        candidates: list[Citation],
        admissible: AdmissibleSet,
        run_id: str = "adhoc",
        k: int = 8,
    ) -> AnswerResult:
        started = time.perf_counter()

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000

        verdict = pre_flight(trace)
        if not verdict.passed:
            assert verdict.reason is not None
            return Refusal(
                query=trace.query,
                as_of=trace.as_of,
                reason=verdict.reason,
                detail=verdict.detail,
                trace=trace,
                cost_usd=Decimal(0),
                latency_ms=elapsed(),
            )

        shown = self._reranker.rerank(trace.query, candidates, k=k)
        request = CompletionRequest(
            model=self._model,
            system=SYSTEM_PROMPT,
            messages=(Message(Role.USER, build_user_message(trace.query, trace.as_of, shown)),),
            max_tokens=self._max_tokens,
            json_schema=RESPONSE_SCHEMA,
            run_id=run_id,
        )

        try:
            response = await self._llm.complete(request)
        except CompletionError as exc:
            # An exception is NOT a refusal. Let it propagate; the API layer
            # turns it into a 502. Rendering an outage as "we decided not to
            # answer" would be a lie told by the error path.
            log.error("answer.provider_unavailable", error=str(exc))
            raise

        cost = response.cost_usd

        def refuse(reason: RefusalReason, detail: str) -> Refusal:
            return Refusal(
                query=trace.query,
                as_of=trace.as_of,
                reason=reason,
                detail=detail,
                trace=trace,
                cost_usd=cost,
                latency_ms=elapsed(),
            )

        # Check the stop reason before touching the text.
        if response.refused:
            return refuse(
                RefusalReason.PROVIDER_REFUSED,
                "The model declined to answer this question.",
            )

        try:
            payload = json.loads(response.text)
            sufficient = bool(payload["sufficient"])
            text = str(payload["answer"])
            cited_ids = [int(i) for i in payload["citation_ids"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning("answer.malformed", error=str(exc))
            return refuse(
                RefusalReason.MALFORMED_RESPONSE,
                "The model's response could not be read.",
            )

        if not sufficient or not text.strip():
            return refuse(
                RefusalReason.INSUFFICIENT_EVIDENCE,
                (f"The provisions in force on {trace.as_of} do not cover this question."),
            )

        result = validate_citations(
            cited_ids,
            retrieved_ids={c.chunk_id for c in shown},
            admissible=admissible,
        )
        if not result.ok:
            # Note what is NOT logged: the answer text. It is discarded here
            # and must not leak into a log that someone later treats as a
            # record of what the system said.
            log.error(
                "answer.fabricated_citation",
                query=trace.query[:80],
                failures=[f"{f.chunk_id}:{f.check}" for f in result.failures],
            )
            return refuse(
                RefusalReason.FABRICATED_CITATION,
                (
                    "The generated answer cited a source that could not be "
                    "verified, so it was discarded."
                ),
            )

        by_id = {c.chunk_id: c for c in shown}
        citations = tuple(by_id[i] for i in result.cited)

        log.info(
            "answer.ok",
            run_id=run_id,
            citations=len(citations),
            latency_ms=round(elapsed(), 1),
        )
        return Answer(
            query=trace.query,
            as_of=trace.as_of,
            text=text,
            citations=citations,
            trace=trace,
            cost_usd=cost,
            latency_ms=elapsed(),
        )
