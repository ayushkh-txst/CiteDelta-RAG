"""The HTTP surface."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from json import dumps
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response, StreamingResponse

from citedelta.answer.intent import Intent, classify
from citedelta.answer.models import Answer, AnswerResult, Citation, Refusal, RefusalReason
from citedelta.answer.resolve import resolve_followup
from citedelta.answer.service import PhaseHook
from citedelta.api.state import AppState, build_state, close_state
from citedelta.api.traces import load_thread, next_turn_index, persist
from citedelta.bench.temporal import load_admissible
from citedelta.config import get_settings
from citedelta.retrieve import RetrievalTrace, hybrid_search
from citedelta.temporal import AdmissibleSet
from citedelta.web.copy import GREETING_REPLY, REFUSAL_HELP, REFUSAL_LABELS
from citedelta.web.diff import diff_pair
from citedelta.web.filters import (
    citation_chips,
    compare_dates,
    highlight_quote,
    markdown_lite,
    ordinal,
    strength,
)
from citedelta.web.transcript import (
    Rupture,
    TurnView,
    amendments_between,
    build_transcript,
)
from substrate.db import Database
from substrate.llm import CompletionError
from substrate.obs import new_run_id

log = structlog.get_logger(__name__)

ASKS = Counter("citedelta_asks_total", "Questions asked", ["outcome"])
LATENCY = Histogram("citedelta_ask_seconds", "End-to-end /ask latency")

router = APIRouter()


@dataclass
class PendingTurn:
    """An in-flight turn and its phase channel.

    Held in process memory, which means this design assumes ONE worker.
    That is already the shape of the deployment (the saturation curve and
    the QPS knee are single-process numbers), so it is a documented
    constraint rather than an accident. Multiple workers would need sticky
    routing or a shared channel — noted in the runbook, not solved here.
    """

    queue: asyncio.Queue[str | None]
    task: asyncio.Task[str]


_PENDING: dict[str, PendingTurn] = {}

WEB = Path(__file__).resolve().parent.parent / "web"
templates = Jinja2Templates(directory=str(WEB / "templates"))
templates.env.filters["citation_chips"] = citation_chips
templates.env.filters["markdown_lite"] = markdown_lite
templates.env.filters["highlight_quote"] = highlight_quote
templates.env.filters["ordinal"] = ordinal
templates.env.filters["strength"] = strength
templates.env.globals["REFUSAL_LABELS"] = REFUSAL_LABELS
templates.env.globals["REFUSAL_HELP"] = REFUSAL_HELP


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    as_of: date | None = None
    k: int = Field(default=8, ge=1, le=20)
    conversation_id: UUID | None = None
    """Optional thread identity. When absent a fresh conversation starts — the
    JSON contract's existing callers are unchanged, and every row still lands
    in a conversation."""

    model_config = {"extra": "forbid"}


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


async def _run_ask(
    state: AppState, body: AskRequest, *, on_phase: PhaseHook | None = None
) -> dict[str, Any]:
    """The one code path behind both /ask and /ui/ask."""
    run_id = new_run_id()
    requested_as_of = body.as_of or datetime.now(UTC).date()

    async def phase(text: str) -> None:
        if on_phase is not None:
            await on_phase(text)

    conversation_id = body.conversation_id or uuid4()
    turn_index = await next_turn_index(state.db, conversation_id)

    # Intercepting here, ahead of admissible-load and embedding, is what makes
    # a greeting actually free. AnswerService guards the same case for callers
    # that reach it directly (the CLI); this is the HTTP path skipping the
    # work entirely.
    if classify(body.query) is Intent.GREETING:
        greeting = Refusal(
            query=body.query,
            as_of=requested_as_of.isoformat(),
            reason=RefusalReason.GREETING,
            detail=GREETING_REPLY,
            trace=None,
            cost_usd=Decimal(0),
            latency_ms=0.0,
        )
        trace_id = await persist(
            state.db,
            result=greeting,
            candidates=[],
            as_of=requested_as_of,
            run_id=run_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
        return {
            "result": greeting,
            "candidates": [],
            "trace_id": trace_id,
            "selectivity": 0.0,
        }

    history = await load_thread(state.db, conversation_id)
    if history:
        await phase("Resolving the question")
    resolution = await resolve_followup(
        state.resolver_llm,
        question=body.query,
        history=history,
        current_as_of=requested_as_of,
        corpus_since=state.corpus_since,
        today=datetime.now(UTC).date(),
        run_id=run_id,
    )
    as_of = resolution.as_of or requested_as_of
    search_text = resolution.standalone_question

    with LATENCY.time():
        await phase(f"Finding provisions in force on {as_of:%d %b %Y}")
        admissible = await load_admissible(as_of, state.corpus_size, db=state.db)
        # The internal label is "valid_on=YYYY-MM-DD" (benchmark-facing). The
        # product-facing label is the date itself, so the banner, the prompt,
        # and the refusal detail read naturally rather than leaking the model.
        admissible = AdmissibleSet(
            ids=admissible.ids,
            label=as_of.isoformat(),
            corpus_size=admissible.corpus_size,
        )
        query_vector = state.embeddings.embed([search_text])[0]
        trace = hybrid_search(
            search_text,
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
            on_phase=phase,
        )
        # The resolver and the answer share a run_id in one ledger; the row's
        # cost is the turn's true cost across both calls, not just the answer.
        if state.ledger.total(run_id) != result.cost_usd:
            result = replace(result, cost_usd=state.ledger.total(run_id))
        trace_id = await persist(
            state.db,
            result=result,
            candidates=candidates,
            as_of=as_of,
            run_id=run_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
            query=body.query,
            resolved_query=search_text,
        )

    ASKS.labels(outcome="refused" if result.refused else "answered").inc()
    return {
        "result": result,
        "candidates": candidates,
        "trace_id": trace_id,
        "selectivity": len(admissible.ids) / admissible.corpus_size,
        "resolved_query": search_text if search_text != body.query else None,
        "as_of": as_of.isoformat(),
    }


@router.post("/ui/ask", response_class=HTMLResponse)
async def ui_ask(
    request: Request,
    query: str = Form(...),
    as_of: date | None = Form(None),  # noqa: B008 - FastAPI form metadata
    conversation_id: UUID | None = Form(None),  # noqa: B008
) -> HTMLResponse:
    """Start the turn, return the shell immediately.

    The shell renders the question straight away — the user sees their own
    words land before any work happens, which is most of what makes an
    interface feel responsive. The question itself is POSTed (never a query
    string), because an EventSource can only GET and these questions carry
    personal circumstances that should not land in access logs or browser
    history.
    """
    state = ctx(request)
    turn_id = uuid4().hex
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def emit(text: str) -> None:
        await queue.put(text)

    async def run() -> str:
        try:
            return await _run_turn_html(request, state, query, as_of, conversation_id, emit)
        finally:
            await queue.put(None)  # close the stream even if the turn raised

    _PENDING[turn_id] = PendingTurn(queue=queue, task=asyncio.create_task(run()))

    pending_date = as_of or _today()
    return templates.TemplateResponse(
        request,
        "partials/turn_pending.html",
        {
            "turn_id": turn_id,
            "question": query,
            "as_of": pending_date.isoformat(),
            "stamp_day": f"{pending_date.day} {pending_date:%b}",
            "stamp_year": f"{pending_date:%Y}",
        },
    )


@router.get("/ui/turn/{turn_id}/stream")
async def turn_stream(turn_id: str) -> StreamingResponse:
    pending = _PENDING.get(turn_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="no such turn")

    async def events() -> AsyncIterator[str]:
        try:
            while True:
                item = await pending.queue.get()
                if item is None:
                    break
                yield f"event: phase\ndata: {dumps(item)}\n\n"
            try:
                html = await pending.task
            except Exception:
                log.exception("turn.failed", turn_id=turn_id)
                html = _error_html()
            # One line, so the SSE frame stays well-formed.
            yield f"event: done\ndata: {dumps(html)}\n\n"
        finally:
            _PENDING.pop(turn_id, None)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _today() -> date:
    return datetime.now(UTC).date()


def _error_html() -> str:
    return (
        '<div class="phase"><span class="mk">·</span>'
        "<span>Something went wrong on our side. Ask again?</span></div>"
    )


def _render_turn_fragment(
    request: Request,
    state: AppState,
    *,
    turn: TurnView,
    candidates: list[Citation],
    cited_ids: list[int],
    max_score: float,
    trace_id: int,
) -> str:
    """One turn's HTML. The rupture prefix is added by the caller, which
    knows the previous turn's date."""
    options = (
        compare_dates(
            list(turn.result.citations), as_of=turn.as_of, corpus_since=state.corpus_since
        )
        if isinstance(turn.result, Answer) and turn.result.citations
        else []
    )
    response = templates.TemplateResponse(
        request,
        "partials/turn.html",
        {
            "turn": turn,
            "candidates": candidates,
            "cited_ids": cited_ids,
            "max_score": max_score,
            "trace_id": trace_id,
            "compare_options": options,
            "corpus_since": state.corpus_since.isoformat(),
        },
    )
    return bytes(response.body).decode()


async def _run_turn_html(
    request: Request,
    state: AppState,
    query: str,
    as_of: date | None,
    conversation_id: UUID | None,
    emit: PhaseHook,
) -> str:
    """Run one turn and render the finished turn's HTML.

    The JSON and HTML paths share `_run_ask`; only the rendering differs.
    `emit` carries the phase feed to the SSE stream. The fragment includes a
    rupture when this turn's as-of differs from the previous turn's, so the
    appended record reads as an event rather than an unexplained new date.
    """
    effective_cid = conversation_id or uuid4()
    body = AskRequest(query=query, as_of=as_of, conversation_id=effective_cid)
    payload = await _run_ask(state, body, on_phase=emit)

    result = payload["result"]
    turn_as_of = date.fromisoformat(payload["as_of"])

    history = await load_thread(state.db, effective_cid, limit=2)
    previous = history[-1] if history else None
    continuation = previous is not None and previous.as_of == turn_as_of

    turn = TurnView(
        question=body.query,
        as_of=turn_as_of,
        result=result,
        resolved_query=payload.get("resolved_query"),
        continuation=continuation,
    )

    prefix = ""
    if previous is not None and previous.as_of != turn_as_of:
        rupture = Rupture(
            as_of=turn_as_of,
            amendments_between=amendments_between(
                previous.as_of, turn_as_of, state.amendment_dates
            ),
            earlier=turn_as_of < previous.as_of,
        )
        prefix = bytes(
            templates.TemplateResponse(request, "partials/rupture.html", {"entry": rupture}).body
        ).decode()

    citations = list(result.citations) if isinstance(result, Answer) else []
    cited_ids = [c.chunk_id for c in citations]
    max_score = max((c.rrf_score for c in payload["candidates"]), default=1.0) or 1.0

    return prefix + _render_turn_fragment(
        request,
        state,
        turn=turn,
        candidates=payload["candidates"],
        cited_ids=cited_ids,
        max_score=max_score,
        trace_id=payload["trace_id"],
    )


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    state = ctx(request)
    today = _today()
    conversation_id = uuid4()
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "transcript_html": await _render_thread_html(request, state, conversation_id),
            "conversation_id": str(conversation_id),
            "as_of": today.isoformat(),
            "as_of_label": today.strftime("%-d %b %Y"),
            "corpus_since": state.corpus_since.isoformat(),
            "today": today.isoformat(),
            "snapshot_count": state.snapshot_count,
        },
    )


@dataclass(frozen=True, slots=True)
class RenderableTurn:
    """A rebuilt turn plus everything its disclosures need to render.

    The disclosure partials (Sources, Compare, Trace) need the full candidate
    list — including uncited rows — which lives in the trace, not on the
    `TurnView`. This bundles them so the transcript can be rendered from the
    database exactly as the live turn was rendered from memory.
    """

    turn: TurnView
    candidates: list[Citation]
    cited_ids: list[int]
    max_score: float
    trace_id: int


async def _render_thread_html(request: Request, state: AppState, conversation_id: UUID) -> Markup:
    """Render every turn of a conversation as one safe HTML fragment."""
    thread = await _turns_for_thread(state.db, conversation_id)
    entries = build_transcript([t.turn for t in thread], amendment_dates=state.amendment_dates)
    by_identity = {id(t.turn): t for t in thread}
    fragments: list[str] = []
    for entry in entries:
        if isinstance(entry, Rupture):
            fragments.append(
                bytes(
                    templates.TemplateResponse(
                        request, "partials/rupture.html", {"entry": entry}
                    ).body
                ).decode()
            )
        else:
            renderable = by_identity[id(entry)]
            fragments.append(
                _render_turn_fragment(
                    request,
                    state,
                    turn=entry,
                    candidates=renderable.candidates,
                    cited_ids=renderable.cited_ids,
                    max_score=renderable.max_score,
                    trace_id=renderable.trace_id,
                )
            )
    return Markup("".join(fragments))  # noqa: S704 - each fragment escaped during rendering


@router.post("/ui/compare", response_class=HTMLResponse)
async def ui_compare(
    request: Request,
    query: str = Form(...),
    left: date = Form(...),  # noqa: B008
    right: date = Form(...),  # noqa: B008
    conversation_id: UUID | None = Form(None),  # noqa: B008
) -> HTMLResponse:
    """Two full runs, deliberately.

    Not one run re-filtered for each date: the two dates have different
    admissible sets, so retrieval genuinely differs. Reusing one result and
    filtering it afterwards is post-filtering — and the temporal benchmarks
    measured what that costs at this corpus's ~2% selectivity.
    """
    state = ctx(request)
    a = await _run_ask(state, AskRequest(query=query, as_of=left))
    b = await _run_ask(state, AskRequest(query=query, as_of=right))

    left_html = right_html = None
    if not a["result"].refused and not b["result"].refused:
        left_html, right_html = diff_pair(a["result"].text, b["result"].text)

    return templates.TemplateResponse(
        request,
        "partials/compare_turn.html",
        {
            "query": query,
            "left_date": left,
            "right_date": right,
            "left_html": left_html,
            "right_html": right_html,
            "left_refused": a["result"].refused,
            "right_refused": b["result"].refused,
        },
    )


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: int, request: Request) -> dict[str, Any]:
    async with ctx(request).db.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM query_traces WHERE id = $1", trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such trace")
    return {k: _jsonable(v) for k, v in dict(row).items()}


async def _turns_for_thread(
    db: Database, conversation_id: UUID, *, limit: int = 40
) -> list[RenderableTurn]:
    """Rebuild the renderable record for a conversation from its traces.

    The trace rows keep the candidates (ids, dates, ranks, and now the
    verified quotes) but only a snippet of each chunk's text, so the full
    provision text is re-hydrated from Postgres to render Sources. This is
    what lets the transcript page show old turns with their disclosures,
    not just the turn that was streamed in live.
    """
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM query_traces
               WHERE conversation_id = $1
               ORDER BY turn_index ASC
               LIMIT $2""",
            conversation_id,
            limit,
        )
        chunk_ids = {int(c["chunk_id"]) for r in rows for c in (r["candidates"] or [])}
        texts: dict[int, str] = {}
        for start in range(0, len(chunk_ids), 500):
            batch = list(chunk_ids)[start : start + 500]
            if not batch:
                continue
            texts.update(
                {
                    int(r["id"]): str(r["text"])
                    for r in await conn.fetch(
                        "SELECT id, text FROM chunks WHERE id = ANY($1::bigint[])",
                        batch,
                    )
                }
            )

    views: list[RenderableTurn] = []
    previous: date | None = None
    for r in rows:
        as_of = r["as_of"]
        candidates = [
            Citation(
                chunk_id=int(c["chunk_id"]),
                citation_path=str(c["citation_path"]),
                effective_from=str(c["effective_from"]),
                effective_to=c["effective_to"],
                text=texts.get(int(c["chunk_id"]), str(c.get("snippet") or "")),
                rrf_score=float(c["rrf_score"]),
                ranks=dict(c.get("ranks") or {}),
                quote=str(c.get("quote") or ""),
            )
            for c in (r["candidates"] or [])
        ]
        cited_ids = list(r["cited_ids"] or [])
        citations = tuple(c for c in candidates if c.chunk_id in cited_ids)
        trace = RetrievalTrace(
            query=str(r["query"]),
            as_of=r["as_of"].isoformat(),
            selectivity=float(r["selectivity"] or 0.0),
            candidates_lexical=int(r["candidates_lexical"] or 0),
            candidates_vector=int(r["candidates_vector"] or 0),
            fused=len(candidates),
            hits=[],
        )
        if r["answer"] is not None:
            result: AnswerResult = Answer(
                query=str(r["query"]),
                as_of=r["as_of"].isoformat(),
                text=str(r["answer"]),
                citations=citations,
                trace=trace,
                cost_usd=r["cost_usd"],
                latency_ms=float(r["latency_ms"] or 0.0),
            )
        else:
            result = Refusal(
                query=str(r["query"]),
                as_of=r["as_of"].isoformat(),
                reason=RefusalReason(str(r["refusal_reason"])),
                detail=str(r["refusal_detail"] or ""),
                trace=trace,
                cost_usd=r["cost_usd"],
                latency_ms=float(r["latency_ms"] or 0.0),
            )
        resolved = r["resolved_query"]
        views.append(
            RenderableTurn(
                turn=TurnView(
                    question=str(r["query"]),
                    as_of=as_of,
                    result=result,
                    resolved_query=str(resolved) if resolved and resolved != r["query"] else None,
                    continuation=previous is not None and previous == as_of,
                ),
                candidates=candidates,
                cited_ids=cited_ids,
                max_score=max((c.rrf_score for c in candidates), default=1.0) or 1.0,
                trace_id=int(r["id"]),
            )
        )
        previous = as_of
    return views


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
