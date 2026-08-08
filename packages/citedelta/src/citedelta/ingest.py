"""Snapshot ingestion. Block 2 version: a for loop.

Block 4 replaces the loop with a job queue and reuses everything else in this
file unchanged. That is the point of writing it this way first.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
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
from substrate.queue import ClaimedJob, JobQueue, JobSpec, Worker

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


QUEUE_NAME = "ingest"
KIND_SNAPSHOT = "snapshot"


async def plan_ingest(dates: list[date] | None = None) -> int:
    """Resolve the work, then enqueue it. Returns the number of jobs created.

    Runs ONCE, in the process you type the command into — not in a worker.
    Discovery is cheap, needs no durability, and doing it here means a worker
    never has to know how to find work, only how to do it.
    """
    settings = get_settings()

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
                source_id,
                EXTERNAL_ID,
                "Nonimmigrant Classes",
                f"{CITATION_PREFIX} Part {PART}",
            )

        records = await client.versions(TITLE, PART)
        timelines = build_timelines(records)
        targets = sorted(dates) if dates else sorted({r.effective_from for r in records})

        specs: list[JobSpec] = []
        for on in targets:
            intervals = [
                iv.model_dump(mode="json")
                for section in timelines
                if (iv := interval_covering(timelines, section, on)) is not None
            ]
            specs.append(
                JobSpec(
                    kind=KIND_SNAPSHOT,
                    queue=QUEUE_NAME,
                    # The natural key for this work is the date, so that IS the
                    # idempotency key. Enqueue twice, get one job.
                    idempotency_key=f"{EXTERNAL_ID}@{on.isoformat()}",
                    max_attempts=4,
                    payload={
                        "on": on.isoformat(),
                        "document_id": document_id,
                        "intervals": intervals,
                    },
                )
            )

        queue = JobQueue(db, queue=QUEUE_NAME)
        created = await queue.enqueue_many(specs)

    log.info("ingest.planned", targets=len(targets), created=created)
    return created


def make_snapshot_handler(
    db: Database, client: ECFRClient, stats: IngestStats
) -> Callable[[ClaimedJob], Awaitable[None]]:
    """Bind long-lived resources once; the queue hands over one job at a time."""

    async def handle(job: ClaimedJob) -> None:
        on = date.fromisoformat(str(job.payload["on"]))
        document_id = int(job.payload["document_id"])
        intervals = [SectionInterval.model_validate(d) for d in job.payload["intervals"]]
        by_section = {iv.section: iv for iv in intervals}

        xml = await client.snapshot_xml(TITLE, PART, on)

        # parse_part is pure CPU on ~1 MB of XML — hundreds of milliseconds.
        # Running it inline would freeze the event loop, stalling every other
        # slot in this worker AND its heartbeats, which is how a healthy worker
        # loses its own leases. to_thread hands it to the thread pool so the
        # loop keeps breathing.
        sections = await asyncio.to_thread(parse_part, xml, citation_prefix=CITATION_PREFIX)

        async with db.acquire() as conn, conn.transaction():
            store = CorpusStore(conn)
            for parsed in sections:
                interval = by_section.get(parsed.section)
                if interval is None:
                    stats.sections_skipped += 1
                    continue
                body = "\n".join(c.text for c in parsed.chunks)
                sv_id, created = await store.insert_section_version(
                    document_id,
                    interval,
                    parsed.heading,
                    sha256(body.encode()).digest(),
                )
                if not created:
                    stats.versions_existing += 1
                    continue
                stats.versions_created += 1
                stats.chunks_written += await store.insert_chunks(sv_id, parsed.chunks)

        stats.snapshots += 1

    return handle


async def run_ingest_worker(*, concurrency: int = 2, drain: bool = True) -> IngestStats:
    settings = get_settings()
    stats = IngestStats()

    async with (
        Database.open(settings.database_url, max_size=concurrency + 4) as db,
        ECFRClient(settings.raw_cache_dir) as client,
    ):
        queue = JobQueue(
            db,
            queue=QUEUE_NAME,
            # Comfortably longer than one snapshot takes; heartbeats extend it
            # for anything slower.
            visibility_timeout=90.0,
            retry_base=1.0,
            retry_cap=30.0,
        )
        worker = Worker(queue, concurrency=concurrency, poll_interval=0.2, heartbeat_interval=15.0)
        worker.install_signal_handlers()
        worker.register(KIND_SNAPSHOT, make_snapshot_handler(db, client, stats))

        if drain:
            await worker.run_until_idle()
        else:
            await worker.run_forever()

    return stats
