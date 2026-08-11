"""The HTTP surface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from typing import Any, cast

import structlog
from fastapi import APIRouter, FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from citedelta.answer.models import Answer, Citation
from citedelta.api.state import AppState, build_state, close_state
from citedelta.api.traces import persist
from citedelta.bench.temporal import load_admissible
from citedelta.config import get_settings
from citedelta.retrieve import RetrievalTrace, hybrid_search
from citedelta.temporal import AdmissibleSet
from substrate.obs import new_run_id

log = structlog.get_logger(__name__)

ASKS = Counter("citedelta_asks_total", "Questions asked", ["outcome"])
LATENCY = Histogram("citedelta_ask_seconds", "End-to-end /ask latency")

router = APIRouter()


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    as_of: date | None = None
    k: int = Field(default=8, ge=1, le=20)


class AskResponse(BaseModel):
    trace_id: int
    refused: bool
    as_of: str
    answer: str | None = None
    refusal_reason: str | None = None
    refusal_detail: str | None = None
    citations: list[dict[str, Any]] = []
    latency_ms: float
    cost_usd: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.ctx = await build_state(settings)
    log.info("api.ready", corpus_size=app.state.ctx.corpus_size)
    try:
        yield
    finally:
        await close_state(app.state.ctx)


def ctx(request: Request) -> AppState:
    return cast(AppState, request.app.state.ctx)


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    """Liveness AND readiness. It touches the database, because an API that
    reports healthy while its pool is dead is worse than one that reports
    nothing."""
    state = ctx(request)
    async with state.db.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return {"status": "ok", "corpus_size": state.corpus_size}


@router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    state = ctx(request)
    run_id = new_run_id()
    as_of = body.as_of or datetime.now(UTC).date()

    with LATENCY.time():
        admissible = await load_admissible(as_of, state.corpus_size)
        # The internal label is "valid_on=YYYY-MM-DD" (benchmark-facing). The
        # product-facing label is the date itself, so the banner, the prompt,
        # and the refusal detail read naturally rather than leaking the model.
        admissible = AdmissibleSet(
            ids=admissible.ids,
            label=as_of.isoformat(),
            corpus_size=admissible.corpus_size,
        )
        query_vector = state.embeddings.embed([body.query])[0]
        trace = hybrid_search(
            body.query,
            query_vector,
            lexical=state.lexical,
            vector=state.vector,
            admissible=admissible,
            k=body.k,
        )
        candidates = await _hydrate(state, trace)
        result = await state.answers.answer(
            trace=trace,
            candidates=candidates,
            admissible=admissible,
            run_id=run_id,
            k=body.k,
        )
        trace_id = await persist(
            state.db,
            result=result,
            candidates=candidates,
            as_of=as_of,
            run_id=run_id,
        )

    ASKS.labels(outcome="refused" if result.refused else "answered").inc()

    if isinstance(result, Answer):
        return AskResponse(
            trace_id=trace_id,
            refused=False,
            as_of=result.as_of,
            answer=result.text,
            citations=[
                {
                    "chunk_id": c.chunk_id,
                    "citation_path": c.citation_path,
                    "in_force": c.in_force_label,
                    "text": c.text,
                }
                for c in result.citations
            ],
            latency_ms=result.latency_ms,
            cost_usd=str(result.cost_usd),
        )

    return AskResponse(
        trace_id=trace_id,
        refused=True,
        as_of=result.as_of,
        refusal_reason=result.reason.value,
        refusal_detail=result.detail,
        latency_ms=result.latency_ms,
        cost_usd=str(result.cost_usd),
    )


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: int, request: Request) -> dict[str, Any]:
    async with ctx(request).db.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM query_traces WHERE id = $1", trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such trace")
    return {k: _jsonable(v) for k, v in dict(row).items()}


async def _hydrate(state: AppState, trace: RetrievalTrace) -> list[Citation]:
    """Fused hits carry ids and scores; the text and dates live in Postgres."""
    ids = [h.chunk_id for h in trace.hits]
    if not ids:
        return []
    async with state.db.acquire() as conn:
        rows = {
            int(r["id"]): r
            for r in await conn.fetch(
                """SELECT c.id, c.citation_path, sv.effective_from,
                          sv.effective_to, c.text
                   FROM chunks c JOIN section_versions sv
                     ON sv.id = c.section_version_id
                   WHERE c.id = ANY($1::bigint[])""",
                ids,
            )
        }
    return [
        Citation(
            chunk_id=h.chunk_id,
            citation_path=str(rows[h.chunk_id]["citation_path"]),
            effective_from=rows[h.chunk_id]["effective_from"].isoformat(),
            effective_to=(
                rows[h.chunk_id]["effective_to"].isoformat()
                if rows[h.chunk_id]["effective_to"]
                else None
            ),
            text=str(rows[h.chunk_id]["text"]),
            rrf_score=h.score,
            ranks=dict(h.ranks),
        )
        for h in trace.hits
        if h.chunk_id in rows
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "quantize"):  # Decimal
        return str(value)
    return value


def create_app() -> FastAPI:
    app = FastAPI(title="CiteDelta", lifespan=lifespan)
    app.include_router(router)
    return app
