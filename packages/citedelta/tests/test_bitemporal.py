"""The schema's promises, asserted."""

from __future__ import annotations

from datetime import UTC, date, datetime
from hashlib import sha256

import asyncpg
import pytest

from citedelta.store.corpus import CorpusStore
from citedelta.temporal import AsOf
from substrate.db import Database

pytestmark = pytest.mark.integration


async def _seed_document(conn: asyncpg.Connection) -> int:
    store = CorpusStore(conn)
    source_id = await store.upsert_source("ecfr", "eCFR", "https://www.ecfr.gov")
    return await store.upsert_document(
        source_id, "title-8/part-214", "Nonimmigrant Classes", "8 CFR Part 214"
    )


async def _add_version(
    conn: asyncpg.Connection,
    document_id: int,
    section: str,
    eff_from: date,
    eff_to: date | None,
    text: str,
    *,
    issue_date: date | None = None,
) -> int:
    digest = sha256(text.encode()).digest()
    sv_id = await conn.fetchval(
        """
        INSERT INTO section_versions
            (document_id, section, heading, effective_from, effective_to,
             issue_date, content_sha256)
        VALUES ($1, $2, '', $3, $4, $5, $6)
        RETURNING id
        """,
        document_id,
        section,
        eff_from,
        eff_to,
        issue_date or eff_from,
        digest,
    )
    await conn.execute(
        """
        INSERT INTO chunks
            (section_version_id, ordinal, citation_path, text,
             char_count, token_count, content_sha256)
        VALUES ($1, 0, $2, $3, $4, $5, $6)
        """,
        sv_id,
        f"8 CFR {section}(a)",
        text,
        len(text),
        len(text.split()),
        digest,
    )
    return int(sv_id)


async def test_asof_returns_the_version_in_force(clean_db: Database) -> None:
    async with clean_db.acquire() as conn, conn.transaction():
        doc = await _seed_document(conn)
        await _add_version(conn, doc, "214.2", date(2017, 1, 18), date(2020, 10, 2), "OLD RULE")
        await _add_version(conn, doc, "214.2", date(2020, 10, 2), None, "NEW RULE")

        store = CorpusStore(conn)

        during_old = await store.chunks_as_of(doc, AsOf(valid_on=date(2019, 6, 1)))
        assert [c.text for c in during_old] == ["OLD RULE"]

        during_new = await store.chunks_as_of(doc, AsOf(valid_on=date(2024, 1, 1)))
        assert [c.text for c in during_new] == ["NEW RULE"]


async def test_boundary_day_belongs_to_the_new_version(clean_db: Database) -> None:
    """Half-open intervals: the changeover day is the NEW rule's first day."""
    async with clean_db.acquire() as conn, conn.transaction():
        doc = await _seed_document(conn)
        await _add_version(conn, doc, "214.2", date(2017, 1, 18), date(2020, 10, 2), "OLD RULE")
        await _add_version(conn, doc, "214.2", date(2020, 10, 2), None, "NEW RULE")

        store = CorpusStore(conn)
        on_the_day = await store.chunks_as_of(doc, AsOf(valid_on=date(2020, 10, 2)))
        assert [c.text for c in on_the_day] == ["NEW RULE"]

        day_before = await store.chunks_as_of(doc, AsOf(valid_on=date(2020, 10, 1)))
        assert [c.text for c in day_before] == ["OLD RULE"]


async def test_before_the_corpus_starts_returns_nothing(clean_db: Database) -> None:
    async with clean_db.acquire() as conn, conn.transaction():
        doc = await _seed_document(conn)
        await _add_version(conn, doc, "214.2", date(2017, 1, 18), None, "RULE")

        store = CorpusStore(conn)
        assert await store.chunks_as_of(doc, AsOf(valid_on=date(2015, 1, 1))) == []


async def test_overlapping_versions_are_rejected_by_the_database(clean_db: Database) -> None:
    """The invariant is structural, not a convention ingestion is trusted to keep."""
    async with clean_db.acquire() as conn:
        async with conn.transaction():
            doc = await _seed_document(conn)
            await _add_version(conn, doc, "214.2", date(2017, 1, 18), date(2021, 1, 1), "A")

        with pytest.raises(asyncpg.ExclusionViolationError):
            async with conn.transaction():
                # starts before the previous one ends
                await _add_version(conn, doc, "214.2", date(2020, 10, 2), None, "B")


async def test_transaction_time_hides_facts_not_yet_recorded(clean_db: Database) -> None:
    """The §214.1 case: effective 2017-01-18, not recorded until 2018-12-22."""
    async with clean_db.acquire() as conn, conn.transaction():
        doc = await _seed_document(conn)

        # PRIOR starts open-ended and is superseded the moment AMENDED is recorded.
        # Backdate its own record first so it was knowable at query time.
        await _add_version(conn, doc, "214.1", date(2016, 12, 23), None, "PRIOR TEXT")
        await conn.execute(
            """
            UPDATE section_versions
            SET recorded_at = $2, superseded_at = $3
            WHERE document_id = $1 AND section = '214.1' AND superseded_at IS NULL
            """,
            doc,
            datetime(2016, 12, 23, tzinfo=UTC),
            datetime(2018, 12, 22, tzinfo=UTC),
        )
        late_id = await _add_version(
            conn,
            doc,
            "214.1",
            date(2017, 1, 18),
            None,
            "AMENDED TEXT",
            issue_date=date(2018, 12, 22),
        )
        # we only learned of the amendment on its issue date
        await conn.execute(
            "UPDATE section_versions SET recorded_at = $2 WHERE id = $1",
            late_id,
            datetime(2018, 12, 22, tzinfo=UTC),
        )

        store = CorpusStore(conn)

        # as it turned out:
        assert [c.text for c in await store.chunks_as_of(doc, AsOf(valid_on=date(2018, 1, 1)))] == [
            "AMENDED TEXT"
        ]

        # as anyone could have known at the time:
        assert [
            c.text
            for c in await store.chunks_as_of(
                doc, AsOf(valid_on=date(2018, 1, 1), known_at=datetime(2018, 1, 1, tzinfo=UTC))
            )
        ] == ["PRIOR TEXT"]
