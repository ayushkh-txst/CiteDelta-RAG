"""Writing and reading query_traces. One row per turn, never updated."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import UUID, uuid4

from citedelta.answer.models import Answer, AnswerResult, Citation
from substrate.db import Database


@dataclass(frozen=True, slots=True)
class PriorTurn:
    """Just enough history to resolve a follow-up.

    Deliberately not the whole trace. The resolver needs to know what was
    asked and what date it was asked about; handing it citations and scores
    would cost tokens for context it cannot use.
    """

    query: str
    resolved_query: str
    as_of: date
    answered: bool


async def load_thread(db: Database, conversation_id: UUID, *, limit: int = 6) -> list[PriorTurn]:
    """Most recent turns, oldest first.

    `limit` exists because the resolver's prompt grows with history and its
    usefulness does not — a follow-up refers to the last turn or two, never
    to turn 1 of 40.
    """
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """SELECT query, resolved_query, as_of, answer IS NOT NULL AS answered
                 FROM query_traces
                WHERE conversation_id = $1
                ORDER BY turn_index DESC
                LIMIT $2""",
            conversation_id,
            limit,
        )
    return [
        PriorTurn(
            query=str(r["query"]),
            resolved_query=str(r["resolved_query"] or r["query"]),
            as_of=r["as_of"],
            answered=bool(r["answered"]),
        )
        for r in reversed(rows)
    ]


async def next_turn_index(db: Database, conversation_id: UUID) -> int:
    async with db.acquire() as conn:
        current = await conn.fetchval(
            "SELECT max(turn_index) FROM query_traces WHERE conversation_id = $1",
            conversation_id,
        )
    return 0 if current is None else int(current) + 1


def _candidate_rows(candidates: list[Citation]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": c.chunk_id,
            "citation_path": c.citation_path,
            "effective_from": c.effective_from,
            "effective_to": c.effective_to,
            "rrf_score": c.rrf_score,
            "ranks": c.ranks,
            "snippet": c.text[:400],
            # The verified verbatim quote, so a transcript re-rendered from
            # the trace can still show the bolded span in Sources.
            "quote": c.quote,
        }
        for c in candidates
    ]


async def persist(
    db: Database,
    *,
    result: AnswerResult,
    candidates: list[Citation],
    as_of: date,
    run_id: str,
    conversation_id: UUID | None = None,
    turn_index: int | None = None,
    query: str | None = None,
    resolved_query: str | None = None,
) -> int:
    trace = result.trace

    # `result.query` is the text retrieval ran on — the RESOLVED question for
    # a follow-up. The row's `query` column must stay what the user actually
    # said, or every follow-up trace becomes a lie about the conversation.
    raw_query = query or result.query

    # Every row belongs to a conversation. Defaults keep callers that have no
    # thread (the JSON path, the eval) working unchanged while satisfying the
    # NOT NULL column — they just start a fresh one-turn conversation.
    conversation_id = conversation_id or uuid4()
    turn_index = 0 if turn_index is None else turn_index

    if isinstance(result, Answer):
        cited_ids: list[int] = [c.chunk_id for c in result.citations]
        answer_text: str | None = result.text
        refusal_reason: str | None = None
        refusal_detail: str | None = None
    else:
        cited_ids = []
        answer_text = None
        refusal_reason = result.reason.value
        refusal_detail = result.detail

    async with db.acquire() as conn:
        trace_id = await conn.fetchval(
            """
            INSERT INTO query_traces (
                run_id, query, as_of, selectivity,
                candidates_lexical, candidates_vector, candidates,
                cited_ids, answer, refusal_reason, refusal_detail,
                latency_ms, cost_usd, conversation_id, turn_index, resolved_query
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            RETURNING id
            """,
            run_id,
            raw_query,
            as_of,
            trace.selectivity if trace else 0.0,
            trace.candidates_lexical if trace else 0,
            trace.candidates_vector if trace else 0,
            # Pass the list, not a pre-dumped string: Database's jsonb codec
            # encodes any param value, so json.dumps() here would double-encode
            # and store a JSON string instead of an array.
            _candidate_rows(candidates),
            cited_ids,
            answer_text,
            refusal_reason,
            refusal_detail,
            result.latency_ms,
            result.cost_usd,
            conversation_id,
            turn_index,
            resolved_query,
        )
    return int(trace_id)
