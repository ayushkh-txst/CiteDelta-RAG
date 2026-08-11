"""Writing the trace. One row, one query, never updated."""

from __future__ import annotations

from datetime import date
from typing import Any

from citedelta.answer.models import Answer, AnswerResult, Citation
from substrate.db import Database


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
) -> int:
    trace = result.trace

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
                latency_ms, cost_usd
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11,$12,$13)
            RETURNING id
            """,
            run_id,
            result.query,
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
        )
    return int(trace_id)
