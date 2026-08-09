"""Embed the corpus once, cached by content hash."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from citedelta.config import get_settings
from citedelta.embed.base import EmbeddingProvider, Vectors
from citedelta.embed.local import LocalEmbeddings
from substrate.db import Database

log = structlog.get_logger(__name__)


@dataclass
class EmbedStats:
    chunks_total: int = 0
    distinct_texts: int = 0
    already_cached: int = 0
    newly_embedded: int = 0

    @property
    def dedup_ratio(self) -> float:
        """How much work the content-hash cache saved."""
        if not self.distinct_texts:
            return 0.0
        return self.chunks_total / self.distinct_texts


async def embed_corpus(
    provider: EmbeddingProvider | None = None,
    *,
    batch_size: int = 64,
    commit_every: int = 512,
) -> EmbedStats:
    """Embed every distinct chunk text that isn't cached yet.

    Not a queue job, on purpose. The queue exists to make work survive worker
    loss and to spread it across processes. This work is one CPU-bound
    process, and it is ALREADY resumable — the cache table is the checkpoint.
    """
    settings = get_settings()
    provider = provider or LocalEmbeddings()
    stats = EmbedStats()

    async with Database.open(settings.database_url) as db:
        async with db.acquire() as conn:
            stats.chunks_total = await conn.fetchval("SELECT count(*) FROM chunks") or 0

            # DISTINCT ON collapses the duplicates: one representative text per
            # content hash. The LEFT JOIN leaves only what isn't cached, which
            # is what makes an interrupted run resume for free.
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (c.content_sha256) c.content_sha256, c.text
                FROM chunks c
                LEFT JOIN embeddings e
                       ON e.content_sha256 = c.content_sha256
                      AND e.model_id = $1
                WHERE e.content_sha256 IS NULL
                ORDER BY c.content_sha256
                """,
                provider.model_id,
            )
            stats.distinct_texts = (
                await conn.fetchval("SELECT count(DISTINCT content_sha256) FROM chunks") or 0
            )

        stats.already_cached = stats.distinct_texts - len(rows)
        log.info(
            "embed.plan",
            chunks=stats.chunks_total,
            distinct=stats.distinct_texts,
            cached=stats.already_cached,
            todo=len(rows),
            dedup_ratio=round(stats.dedup_ratio, 2),
        )

        pending: list[tuple[str, bytes, int, bytes]] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = provider.embed([str(r["text"]) for r in batch], batch_size=batch_size)

            for row, vec in zip(batch, vectors, strict=True):
                pending.append(
                    (
                        provider.model_id,
                        bytes(row["content_sha256"]),
                        provider.dimensions,
                        vec.astype(np.float32).tobytes(),
                    )
                )

            stats.newly_embedded += len(batch)

            # Commit periodically rather than once at the end: a crash should
            # cost minutes, not the whole run.
            if len(pending) >= commit_every or start + batch_size >= len(rows):
                async with db.acquire() as conn, conn.transaction():
                    await conn.executemany(
                        """
                        INSERT INTO embeddings (model_id, content_sha256, dim, vector)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (model_id, content_sha256) DO NOTHING
                        """,
                        pending,
                    )
                pending.clear()
                log.info(
                    "embed.progress",
                    done=stats.newly_embedded,
                    total=len(rows),
                    pct=round(100 * stats.newly_embedded / max(len(rows), 1), 1),
                )

    return stats


async def load_corpus_vectors(
    model_id: str = "BAAI/bge-small-en-v1.5",
) -> tuple[np.ndarray, Vectors]:
    """Every chunk's id and vector, aligned row-for-row.

    Note this returns one row per CHUNK, not per distinct text — so duplicate
    vectors appear repeatedly. That is correct: each chunk is a separately
    citable unit with its own validity interval, and the temporal filter has
    to be able to reach every one of them.
    """
    settings = get_settings()
    async with Database.open(settings.database_url) as db, db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, e.vector, e.dim
            FROM chunks c
            JOIN embeddings e
              ON e.content_sha256 = c.content_sha256 AND e.model_id = $1
            ORDER BY c.id
            """,
            model_id,
        )

    if not rows:
        msg = f"no embeddings for {model_id!r} — run `citedelta embed run` first"
        raise RuntimeError(msg)

    dim = int(rows[0]["dim"])
    ids = np.fromiter((int(r["id"]) for r in rows), dtype=np.int64, count=len(rows))
    vectors = np.empty((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        vectors[i] = np.frombuffer(r["vector"], dtype=np.float32, count=dim)
    return ids, vectors
