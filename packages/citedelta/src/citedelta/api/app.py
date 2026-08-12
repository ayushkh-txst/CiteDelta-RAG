"""The HTTP surface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import structlog
from fastapi import APIRouter, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
from citedelta.web.copy import REFUSAL_HELP, REFUSAL_LABELS
from citedelta.web.diff import diff_pair
from citedelta.web.filters import citation_chips
from citedelta.web.ribbon import build_ribbon
from substrate.llm import CompletionError
from substrate.obs import new_run_id

log = structlog.get_logger(__name__)

ASKS = Counter("citedelta_asks_total", "Questions asked", ["outcome"])
LATENCY = Histogram("citedelta_ask_seconds", "End-to-end /ask latency")

router = APIRouter()

WEB = Path(__file__).resolve().parent.parent / "web"
templates = Jinja2Templates(directory=str(WEB / "templates"))
templates.env.filters["citation_chips"] = citation_chips
templates.env.globals["REFUSAL_LABELS"] = REFUSAL_LABELS
templates.env.globals["REFUSAL_HELP"] = REFUSAL_HELP

EXAMPLE_QUERIES = (
    "Can an F-1 student transfer to another school?",
    "What is the grace period after F-1 program completion?",
    "What is the duration of a student's practical training?",
)


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


class SearchHit(BaseModel):
    chunk_id: int
    score: float
    ranks: dict[str, int]


class SearchResponse(BaseModel):
    as_of: str
    selectivity: float
    hits: list[SearchHit]


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


@router.post("/search", response_model=SearchResponse)
async def search(body: AskRequest, request: Request) -> SearchResponse:
    """Retrieval only — no generation. This is the path with a latency SLO.

    Separate from /ask rather than a flag on it, because the two have
    genuinely different performance characteristics and different targets.
    A single endpoint whose p95 depends on a boolean is an endpoint whose
    dashboard means nothing.
    """
    state = ctx(request)
    as_of = body.as_of or datetime.now(UTC).date()
    admissible = await load_admissible(as_of, state.corpus_size, db=state.db)
    query_vector = state.embeddings.embed([body.query])[0]
    trace = hybrid_search(
        body.query,
        query_vector,
        lexical=state.lexical,
        vector=state.vector,
        admissible=admissible,
        k=body.k,
    )
    return SearchResponse(
        as_of=as_of.isoformat(),
        selectivity=trace.selectivity,
        hits=[SearchHit(chunk_id=h.chunk_id, score=h.score, ranks=h.ranks) for h in trace.hits],
    )


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    state = ctx(request)
    payload = await _run_ask(state, body)
    result = payload["result"]

    if isinstance(result, Answer):
        return AskResponse(
            trace_id=payload["trace_id"],
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
        trace_id=payload["trace_id"],
        refused=True,
        as_of=result.as_of,
        refusal_reason=result.reason.value,
        refusal_detail=result.detail,
        latency_ms=result.latency_ms,
        cost_usd=str(result.cost_usd),
    )


async def _run_ask(state: AppState, body: AskRequest) -> dict[str, Any]:
    """The one code path behind both /ask and /ui/ask."""
    run_id = new_run_id()
    as_of = body.as_of or datetime.now(UTC).date()

    with LATENCY.time():
        admissible = await load_admissible(as_of, state.corpus_size, db=state.db)
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
    return {
        "result": result,
        "candidates": candidates,
        "trace_id": trace_id,
        "selectivity": len(admissible.ids) / admissible.corpus_size,
    }


def _presets(today: date, corpus_since: date) -> list[tuple[str, str]]:
    return [
        ("Today", today.isoformat()),
        ("2019", "2019-06-01"),
        # "2016" must land inside the corpus, not in the gap before it — a
        # preset that always refused would look like a bug.
        ("2016", max(date(2016, 6, 1), corpus_since).isoformat()),
    ]


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    state = ctx(request)
    today = datetime.now(UTC).date()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "corpus_size": f"{state.corpus_size:,}",
            "as_of": today.isoformat(),
            "today": today.isoformat(),
            "min_as_of": state.corpus_since.isoformat(),
            "presets": _presets(today, state.corpus_since),
            "query": None,
            "example_queries": EXAMPLE_QUERIES,
            "snapshot_count": state.snapshot_count,
        },
    )


@router.post("/ui/ask", response_class=HTMLResponse)
async def ui_ask(
    request: Request,
    query: str = Form(...),
    as_of: date | None = Form(None),  # noqa: B008 - FastAPI form metadata
) -> HTMLResponse:
    """The HTML twin of POST /ask.

    Deliberately a separate route rather than content-negotiation on /ask.
    The JSON API and the HTML UI have genuinely different contracts — the API
    returns a trace_id, the UI returns rendered candidates — and one handler
    branching on an Accept header to serve two contracts is how endpoints
    become unmaintainable.
    """
    state = ctx(request)
    body = AskRequest(query=query, as_of=as_of)
    payload = await _run_ask(state, body)

    result = payload["result"]
    as_of_date = body.as_of or datetime.now(UTC).date()
    citations = list(result.citations) if isinstance(result, Answer) else []
    ribbon = build_ribbon(citations, as_of=as_of_date)
    cited_ids = [c.chunk_id for c in citations]
    cited_index = {c.chunk_id: i + 1 for i, c in enumerate(citations)}
    max_score = max((c.rrf_score for c in payload["candidates"]), default=1.0) or 1.0

    return templates.TemplateResponse(
        request,
        "partials/answer.html",
        {
            "result": result,
            "selectivity": payload["selectivity"],
            "candidate_count": len(payload["candidates"]),
            "candidates": payload["candidates"],
            "trace_id": payload["trace_id"],
            "ribbon": ribbon,
            "cited_ids": cited_ids,
            "cited_index": cited_index,
            "max_score": max_score,
            "corpus_since": state.corpus_since.isoformat(),
        },
    )


@router.get("/compare", response_class=HTMLResponse)
async def compare(
    request: Request,
    query: str,
    left: date,
    right: date,
) -> HTMLResponse:
    """Same question, two dates. Two full runs, deliberately.

    Not a cached re-render of one run: the two dates have different admissible
    sets, so retrieval genuinely differs. Reusing one result and re-filtering
    it afterwards would be post-filtering — the exact mistake the temporal
    benchmarks measured the cost of.
    """
    state = ctx(request)
    a = await _run_ask(state, AskRequest(query=query, as_of=left))
    b = await _run_ask(state, AskRequest(query=query, as_of=right))

    left_html = right_html = None
    if not a["result"].refused and not b["result"].refused:
        left_html, right_html = diff_pair(a["result"].text, b["result"].text)

    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "corpus_size": f"{state.corpus_size:,}",
            "query": query,
            "left": a,
            "right": b,
            "left_date": left.isoformat(),
            "right_date": right.isoformat(),
            "left_html": left_html,
            "right_html": right_html,
        },
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
    app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")

    @app.exception_handler(CompletionError)
    async def completion_error_handler(request: Request, exc: CompletionError) -> Response:
        # An LLM outage is NOT a refusal and NOT a code bug. Rendering it as
        # either would let the refusal rate absorb the incident and keep the
        # uptime graph green through an outage. A distinct 502 keeps the two
        # failure modes separable.
        log.error("api.provider_unavailable", error=str(exc))
        return Response(
            status_code=502,
            media_type="application/json",
            content='{"detail": "model provider unavailable"}',
        )

    return app
