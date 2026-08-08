"""Reads and writes against the bitemporal corpus. Raw SQL, on purpose."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import asyncpg

from citedelta.ecfr.models import ParsedChunk, SectionInterval
from citedelta.temporal import AsOf

# Shared by every temporal read. $1 = document_id, $2 = valid_on, $3 = known_at.
_ASOF_PREDICATE = """
    sv.document_id = $1
    AND daterange(sv.effective_from, sv.effective_to, '[)') @> $2::date
    AND (
            $3::timestamptz IS NULL
        AND sv.superseded_at IS NULL
        OR  $3::timestamptz IS NOT NULL
        AND sv.recorded_at <= $3::timestamptz
        AND (sv.superseded_at IS NULL OR sv.superseded_at > $3::timestamptz)
    )
    AND NOT sv.removed
"""


@dataclass(frozen=True, slots=True)
class ChunkRow:
    id: int
    section: str
    citation_path: str
    text: str
    effective_from: Any
    effective_to: Any


class CorpusStore:
    def __init__(self, conn: asyncpg.Connection[Any]) -> None:
        self._conn = conn

    async def upsert_source(self, slug: str, name: str, base_url: str) -> int:
        row = await self._conn.fetchrow(
            """
            INSERT INTO sources (slug, name, base_url)
            VALUES ($1, $2, $3)
            ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            slug,
            name,
            base_url,
        )
        return int(row["id"])

    async def upsert_document(
        self, source_id: int, external_id: str, title: str, citation: str
    ) -> int:
        row = await self._conn.fetchrow(
            """
            INSERT INTO documents (source_id, external_id, title, citation)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (source_id, external_id)
                DO UPDATE SET title = EXCLUDED.title
            RETURNING id
            """,
            source_id,
            external_id,
            title,
            citation,
        )
        return int(row["id"])

    async def chunks_as_of(
        self,
        document_id: int,
        as_of: AsOf,
        *,
        citation_prefix: str | None = None,
        limit: int | None = None,
    ) -> list[ChunkRow]:
        """Every chunk in force at a point in bitemporal space."""
        sql = f"""
            SELECT c.id, sv.section, c.citation_path, c.text,
                   sv.effective_from, sv.effective_to
            FROM chunks c
            JOIN section_versions sv ON sv.id = c.section_version_id
            WHERE {_ASOF_PREDICATE}
              AND ($4::text IS NULL OR c.citation_path LIKE $4 || '%')
            ORDER BY sv.section, c.ordinal
            {"LIMIT " + str(int(limit)) if limit else ""}
        """  # noqa: S608 - constant predicate; limit is int()'d before
        rows = await self._conn.fetch(
            sql, document_id, as_of.valid_on, as_of.known_at, citation_prefix
        )
        return [
            ChunkRow(
                id=r["id"],
                section=r["section"],
                citation_path=r["citation_path"],
                text=r["text"],
                effective_from=r["effective_from"],
                effective_to=r["effective_to"],
            )
            for r in rows
        ]

    async def count_as_of(self, document_id: int, as_of: AsOf) -> int:
        row = await self._conn.fetchrow(
            f"""
            SELECT count(*) AS n
            FROM chunks c
            JOIN section_versions sv ON sv.id = c.section_version_id
            WHERE {_ASOF_PREDICATE}
            """,  # noqa: S608
            document_id,
            as_of.valid_on,
            as_of.known_at,
        )
        return int(row["n"])

    async def insert_section_version(
        self,
        document_id: int,
        interval: SectionInterval,
        heading: str,
        content_sha256: bytes,
    ) -> tuple[int, bool]:
        """Insert if absent. Returns (id, created).

        ON CONFLICT DO NOTHING against the partial unique index — note the
        index predicate has to be repeated in the conflict target. `created`
        is False when the version already exists, so re-running ingestion is a
        no-op rather than a duplicate.
        """
        row = await self._conn.fetchrow(
            """
            INSERT INTO section_versions
                (document_id, section, heading, effective_from, effective_to,
                 issue_date, removed, content_sha256)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (document_id, section, effective_from)
                WHERE superseded_at IS NULL
                DO NOTHING
            RETURNING id
            """,
            document_id,
            interval.section,
            heading,
            interval.effective_from,
            interval.effective_to,
            interval.issue_date,
            interval.removed,
            content_sha256,
        )
        if row is not None:
            return int(row["id"]), True

        existing = await self._conn.fetchval(
            """
            SELECT id FROM section_versions
            WHERE document_id = $1 AND section = $2 AND effective_from = $3
              AND superseded_at IS NULL
            """,
            document_id,
            interval.section,
            interval.effective_from,
        )
        return int(existing), False

    async def insert_chunks(self, section_version_id: int, chunks: list[ParsedChunk]) -> int:
        if not chunks:
            return 0
        await self._conn.executemany(
            """
            INSERT INTO chunks
                (section_version_id, ordinal, citation_path, text,
                 char_count, token_count, content_sha256)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (section_version_id, ordinal) DO NOTHING
            """,
            [
                (
                    section_version_id,
                    c.ordinal,
                    c.citation_path,
                    c.text,
                    len(c.text),
                    len(c.text.split()),  # placeholder; real tokenizer in Block 5
                    sha256(c.text.encode()).digest(),
                )
                for c in chunks
            ],
        )
        return len(chunks)
