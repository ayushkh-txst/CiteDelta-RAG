"""Populate the baseline table from the same vectors the indexes use."""

from __future__ import annotations

import psycopg
import structlog

from citedelta.config import get_settings
from citedelta.embed.corpus import load_corpus_vectors
from citedelta.index.vector import Ids, Vectors

log = structlog.get_logger(__name__)

# Matches the hand-written HNSW/IVF defaults so the comparison measures
# implementations, not hyperparameters.
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 64
DEFAULT_LISTS = 100


def rebuild_baseline(
    ids: Ids,
    vectors: Vectors,
    dsn: str,
    *,
    probe: str = "both",
    m: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
    lists: int = DEFAULT_LISTS,
) -> None:
    """Point `baseline_vectors` at EXACTLY this (ids, vectors) pair.

    The benchmark harness holds 500 vectors out as queries, so the hand-written
    indexes are built on the remaining 37,711 — not the full 38,211. The
    baseline must contain the same rows the hand-written indexes were built
    from, or the comparison measures the split, not the index. pgvector's
    `build()` calls this so both sides always see the same data.

    `probe` selects which index to build:
      * "both" — HNSW and IVFFlat (used by the standalone load_baseline)
      * "hnsw" — HNSW only. The planner prefers HNSW when both exist, so an
        "ivf" measurement would silently use HNSW; each probe must build ONLY
        its own index for the comparison to mean anything.
      * "ivf"  — IVFFlat only, for the same reason.

    psycopg, not asyncpg: asyncpg's binary COPY has no encoder for the
    `vector` type. The benchmark adapter is sync anyway (see pgvector.py),
    so a sync loader keeps the whole baseline path on one driver.

    Indexes are created AFTER the load. Building into an empty index and
    inserting row by row is far slower and is the classic first-benchmark
    error. Parallel maintenance workers would need shared memory (/dev/shm is
    64MB in this container) — a single-worker build uses local memory.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE baseline_vectors")
        with (
            conn.cursor() as cur,
            cur.copy("COPY baseline_vectors (chunk_id, embedding) FROM STDIN") as copy,
        ):
            for cid, vec in zip(ids, vectors, strict=True):
                text = "[" + ",".join(f"{v:.7g}" for v in vec) + "]"
                copy.write_row((int(cid), text))
        log.info("pgvector.loaded", count=len(ids))

        conn.execute("SET maintenance_work_mem = '128MB'")
        conn.execute("SET max_parallel_maintenance_workers = 0")
        conn.execute("DROP INDEX IF EXISTS baseline_hnsw_idx")
        conn.execute("DROP INDEX IF EXISTS baseline_ivf_idx")
        if probe in ("both", "hnsw"):
            conn.execute(
                f"""CREATE INDEX baseline_hnsw_idx ON baseline_vectors
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef_construction})"""
            )
        if probe in ("both", "ivf"):
            conn.execute(
                f"""CREATE INDEX baseline_ivf_idx ON baseline_vectors
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {lists})"""
            )
        log.info(
            "pgvector.indexed",
            probe=probe,
            m=m,
            ef_construction=ef_construction,
            lists=lists,
        )


async def load_baseline(
    *,
    m: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
    lists: int = DEFAULT_LISTS,
) -> None:
    """Load the full corpus and build both pgvector indexes. Used to report
    build time separately; the benchmark itself calls `rebuild_baseline` per
    dataset through the adapter's `build()`."""
    ids, vectors = await load_corpus_vectors()
    rebuild_baseline(
        ids,
        vectors,
        get_settings().database_url,
        probe="both",
        m=m,
        ef_construction=ef_construction,
        lists=lists,
    )
