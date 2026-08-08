"""Chaos #1 — SIGKILL a worker mid-ingest; prove the corpus is unharmed.

Baseline the corpus from an uninterrupted run, wipe it, then ingest again
while hard-killing the worker partway through. The two corpora must be
byte-identical.

Row ids are deliberately NOT compared: identity sequences advance differently
after a rollback, so ids legitimately differ. Content hashes are what has to
match, and content is what anyone actually reads.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time

from citedelta.config import get_settings
from citedelta.ingest import QUEUE_NAME
from substrate.db import Database
from substrate.queue import JobQueue

FINGERPRINT_SQL = """
SELECT count(*) AS n,
       coalesce(md5(string_agg(h, ',' ORDER BY h)), '') AS digest
FROM (SELECT encode(content_sha256, 'hex') AS h FROM chunks) t
"""


async def fingerprint(db: Database) -> tuple[int, str]:
    async with db.acquire() as conn:
        row = await conn.fetchrow(FINGERPRINT_SQL)
    return int(row["n"]), str(row["digest"])


async def reset(db: Database) -> None:
    async with db.acquire() as conn:
        await conn.execute("TRUNCATE jobs, chunks, section_versions RESTART IDENTITY CASCADE")


def run_cli(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "citedelta.cli", *args], check=True)


def spawn_worker() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "citedelta.cli", "work", "-c", "2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def main() -> int:
    settings = get_settings()

    async with Database.open(settings.database_url) as db:
        print("→ baseline: uninterrupted ingest")
        await reset(db)
        run_cli("plan")
        run_cli("work", "-c", "2")
        expected = await fingerprint(db)
        print(f"  {expected[0]} chunks, digest {expected[1][:16]}…")

        queue = JobQueue(db, queue=QUEUE_NAME)

        print("→ chaos: SIGKILL the worker mid-run")
        await reset(db)
        run_cli("plan")

        proc = spawn_worker()
        mid = None
        partial = (0, "")
        deadline = time.time() + 30
        while time.time() < deadline:
            mid = await queue.stats()
            partial = await fingerprint(db)
            # Kill once the worker has INGESTED something but still has work
            # left. This stays correct however warm the disk cache is.
            if partial[0] > 0 and mid.outstanding > 0:
                break
            if mid.outstanding == 0 and partial[0] > 0:
                raise AssertionError("worker finished before we could kill it")
            time.sleep(0.25)
        else:
            raise AssertionError("worker never reached a killable mid-run state")
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        print(f"  killed pid {proc.pid}")

        partial = await fingerprint(db)
        mid = await queue.stats()
        print(f"  mid-run: {mid.model_dump()}  ({partial[0]} chunks so far)")
        assert mid.outstanding > 0, "killed too late — nothing was left to do"
        assert partial[0] > 0, "killed too early — nothing had been ingested"

        print("→ recovery: leases lapse, a new worker picks the work up")
        # The SIGKILLed worker's in-flight jobs still hold a ~90s lease, so a
        # single drain exits while they are still 'running' and they never get
        # redone. Keep working until the leases lapse and every job is terminal.
        queue = JobQueue(db, queue=QUEUE_NAME)
        deadline = time.time() + 180
        while True:
            run_cli("work", "-c", "2")
            final = await queue.stats()
            if final.outstanding == 0:
                break
            if time.time() > deadline:
                raise AssertionError("queue never drained after recovery")
            time.sleep(1)

        actual = await fingerprint(db)
        final = await queue.stats()
        print(f"  {actual[0]} chunks, digest {actual[1][:16]}…  {final.model_dump()}")

        assert actual == expected, f"corpus diverged: {actual} != {expected}"
        assert final.dead == 0, "jobs were dead-lettered"
        assert final.outstanding == 0, "work was left behind"

    print("\n✅ no chunks lost, none duplicated, corpus byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
