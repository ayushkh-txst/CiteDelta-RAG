"""Snapshot ingestion. Block 2 version: a for loop.

Block 4 replaces the loop with a job queue and reuses everything else in this
file unchanged. That is the point of writing it this way first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256

import structlog

from citedelta.config import get_settings
from citedelta.ecfr.client import ECFRClient
from citedelta.ecfr.models import SectionInterval
from citedelta.ecfr.parse import parse_part
from citedelta.ecfr.timeline import build_timelines, interval_covering
from citedelta.store.corpus import CorpusStore
from substrate.db import Database

log = structlog.get_logger(__name__)

TITLE = 8
PART = "214"
CITATION_PREFIX = "8 CFR"
EXTERNAL_ID = f"title-{TITLE}/part-{PART}"


@dataclass
class IngestStats:
    snapshots: int = 0
    versions_created: int = 0
    versions_existing: int = 0
    chunks_written: int = 0
    sections_skipped: int = 0  # no interval covers this date — corpus horizon


async def ingest_snapshot(
    db: Database,
    client: ECFRClient,
    document_id: int,
    timelines: dict[str, list[SectionInterval]],
    on: date,
    stats: IngestStats,
) -> None:
    """Ingest every section as it stood on one date.

    A snapshot is the complete text in force, so ingesting one date yields a
    complete corpus for that date. That makes the operation idempotent and
    order-independent — exactly what a job queue needs from it.
    """
    xml = await client.snapshot_xml(TITLE, PART, on)
    sections = parse_part(xml, citation_prefix=CITATION_PREFIX)

    async with db.acquire() as conn, conn.transaction():
        store = CorpusStore(conn)
        for parsed in sections:
            interval = interval_covering(timelines, parsed.section, on)
            if interval is None:
                stats.sections_skipped += 1
                continue

            body = "\n".join(c.text for c in parsed.chunks)
            sv_id, created = await store.insert_section_version(
                document_id, interval, parsed.heading, sha256(body.encode()).digest()
            )
            if not created:
                stats.versions_existing += 1
                continue

            stats.versions_created += 1
            stats.chunks_written += await store.insert_chunks(sv_id, parsed.chunks)

    stats.snapshots += 1
    log.info(
        "snapshot.ingested",
        date=str(on),
        sections=len(sections),
        created=stats.versions_created,
        chunks=stats.chunks_written,
    )


async def ingest_dates(dates: list[date]) -> IngestStats:
    """The Block 2 driver. One process, one loop, no recovery."""
    settings = get_settings()
    stats = IngestStats()

    async with (
        Database.open(settings.database_url) as db,
        ECFRClient(settings.raw_cache_dir) as client,
    ):
        async with db.acquire() as conn:
            store = CorpusStore(conn)
            source_id = await store.upsert_source(
                "ecfr", "Electronic Code of Federal Regulations", "https://www.ecfr.gov"
            )
            document_id = await store.upsert_document(
                source_id, EXTERNAL_ID, "Nonimmigrant Classes", f"{CITATION_PREFIX} Part {PART}"
            )

        records = await client.versions(TITLE, PART)
        timelines = build_timelines(records)
        log.info("timelines.built", sections=len(timelines), records=len(records))

        for on in sorted(dates):
            await ingest_snapshot(db, client, document_id, timelines, on, stats)

    return stats
