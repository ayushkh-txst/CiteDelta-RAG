from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from citedelta.answer.models import Citation, Refusal, RefusalReason
from citedelta.api.traces import persist
from citedelta.retrieve import RetrievalTrace
from substrate.db import Database


def _citation(chunk_id: int, rank: int) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        citation_path=f"8 CFR 214.2(f)({chunk_id})",
        effective_from="2016-01-01",
        effective_to=None,
        text="text " * 200,
        rrf_score=0.03,
        ranks={"lexical": rank},
    )


@pytest.mark.asyncio
async def test_refusals_are_persisted_too(clean_db: Database) -> None:
    """Refusal rows are how 'how often, and why' gets answered later."""
    trace = RetrievalTrace(query="q", as_of="2019-01-01", selectivity=0.02, fused=0, hits=[])
    result = Refusal(
        query="q",
        as_of="2019-01-01",
        reason=RefusalReason.NO_ADMISSIBLE_SOURCE,
        detail="nothing in force",
        trace=trace,
        cost_usd=Decimal(0),
        latency_ms=12.0,
    )
    trace_id = await persist(
        clean_db, result=result, candidates=[], as_of=date(2019, 1, 1), run_id="r1"
    )
    async with clean_db.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM query_traces WHERE id = $1", trace_id)
    assert row["refusal_reason"] == "no_admissible_source"
    assert row["answer"] is None


@pytest.mark.asyncio
async def test_uncited_candidates_survive_into_the_trace(clean_db: Database) -> None:
    """The whole reason the trace panel is more than a result list."""
    from citedelta.answer.models import Answer

    trace = RetrievalTrace(query="q", as_of="2026-08-11", selectivity=0.02, fused=3, hits=[])
    cited = _citation(1, 1)
    answer = Answer(
        query="q",
        as_of="2026-08-11",
        text="answer [1]",
        citations=(cited,),
        trace=trace,
        cost_usd=Decimal(0),
        latency_ms=100.0,
    )
    candidates = [cited, _citation(2, 2), _citation(3, 3)]
    trace_id = await persist(
        clean_db,
        result=answer,
        candidates=candidates,
        as_of=date(2026, 8, 11),
        run_id="r1",
    )
    async with clean_db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT candidates, cited_ids FROM query_traces WHERE id = $1", trace_id
        )
    stored = row["candidates"]
    assert len(stored) == 3
    assert list(row["cited_ids"]) == [1]
    assert stored[0]["ranks"] == {"lexical": 1}


@pytest.mark.asyncio
async def test_a_row_cannot_be_both_answer_and_refusal(clean_db: Database) -> None:
    """The XOR constraint, asserted. If this ever passes, the invariant has
    quietly been dropped from the schema."""
    import asyncpg

    async with clean_db.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                """INSERT INTO query_traces
                   (run_id, query, as_of, selectivity, candidates_lexical,
                    candidates_vector, candidates, answer, refusal_reason,
                    latency_ms)
                   VALUES ('r','q','2026-01-01',0.1,0,0,'[]'::jsonb,
                           'an answer','also_refused',1.0)"""
            )


@pytest.mark.asyncio
async def test_snippets_are_truncated_so_traces_do_not_store_the_corpus(
    clean_db: Database,
) -> None:
    from citedelta.api.traces import _candidate_rows

    rows = _candidate_rows([_citation(1, 1)])
    assert len(rows[0]["snippet"]) <= 400
