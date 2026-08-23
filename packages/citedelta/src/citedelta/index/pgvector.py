"""pgvector behind the same VectorIndex protocol.

Two adaptations, both worth understanding rather than hiding:

1. SYNC DRIVER. The benchmark harness is synchronous, deliberately — an
   event loop in the measurement path adds scheduling noise to exactly the
   thing being measured. So this adapter uses psycopg rather than asyncpg.
   The app path could use either; the benchmark path should not.

2. MASK → IDS. The protocol speaks boolean masks in internal order, which is
   the right shape for a NumPy index and the wrong shape for a database. The
   conversion is counted inside `search`, not hidden in `compile_filter`,
   because a caller filtering a pgvector index WOULD pay it on every query.
   Moving it out would flatter the baseline.

The `memory_bytes` override returns 0: the index lives in the database's
buffer pool, and pretending the hand-written indexes' in-process resident
size is comparable to Postgres's shared buffers would be a category error.
Build time is reported separately — see `pgvector_load.load_baseline`.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Self

import numpy as np
import psycopg

from citedelta.index.pgvector_load import rebuild_baseline
from citedelta.index.vector import BoolMask, Ids, Neighbor, Vectors


class PgVectorIndex:
    """`effort` maps to `hnsw.ef_search`, matching HNSWIndex's units."""

    def __init__(self, dsn: str, *, probe: str = "hnsw") -> None:
        self._dsn = dsn
        self._probe = probe
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._ids: Ids = np.empty(0, dtype=np.int64)
        self._dim = 512  # placeholder — build() overwrites with vectors.shape[1]

    @property
    def name(self) -> str:
        return f"pgvector-{self._probe}"

    @property
    def size(self) -> int:
        return int(self._ids.size)

    @property
    def dimensions(self) -> int:
        return self._dim

    def build(self, ids: Ids, vectors: Vectors) -> None:
        """Point the baseline table at EXACTLY this (ids, vectors) pair.

        The harness holds 500 vectors out as queries, so the hand-written
        indexes see 37,711 rows, not 38,211. The baseline must see the same
        rows or the comparison measures the split, not the index. The reload
        (COPY + index build) is real work and is timed by the harness as
        build time — the same way the in-process indexes' build is timed.
        """
        rebuild_baseline(ids, vectors, self._dsn, probe=self._probe)
        self._ids = np.asarray(ids, dtype=np.int64)
        self._dim = int(vectors.shape[1])

    def compile_filter(self, admissible_ids: Collection[int]) -> BoolMask:
        wanted = set(map(int, admissible_ids))
        return np.fromiter(
            (int(i) in wanted for i in self._ids), dtype=np.bool_, count=self._ids.size
        )

    def search(
        self,
        query: Vectors,
        k: int,
        *,
        effort: int | None = None,
        admissible: BoolMask | None = None,
    ) -> list[Neighbor]:
        vec = "[" + ",".join(f"{v:.7g}" for v in np.asarray(query).ravel()) + "]"
        ef = max(effort or 40, k)

        with self._conn.cursor() as cur:
            # SET, not SET LOCAL: the connection is autocommit, so a LOCAL
            # setting would be reverted before the next statement — making
            # `effort` a lie. Session-level is fine: every search sets it
            # before running.
            if self._probe == "hnsw":
                cur.execute(f"SET hnsw.ef_search = {ef}")
            else:
                cur.execute(f"SET ivfflat.probes = {effort or 10}")

            if admissible is None:
                cur.execute(
                    """SELECT chunk_id, embedding <=> %s::vector AS d
                       FROM baseline_vectors ORDER BY d LIMIT %s""",
                    (vec, k),
                )
            else:
                allowed = self._ids[admissible].tolist()
                cur.execute(
                    """SELECT chunk_id, embedding <=> %s::vector AS d
                       FROM baseline_vectors
                       WHERE chunk_id = ANY(%s)
                       ORDER BY d LIMIT %s""",
                    (vec, allowed, k),
                )
            return [Neighbor(id=int(r[0]), distance=float(r[1])) for r in cur.fetchall()]

    def memory_bytes(self) -> int:
        return 0

    def save(self, path: Path) -> None:
        """The index lives in Postgres; there is nothing to serialise. Part
        of the honest comparison — persistence and recovery are the
        database's problem, not yours, which is a genuine pgvector win."""

    @classmethod
    def load(cls, path: Path) -> Self:
        raise NotImplementedError("pgvector indexes are not file-backed")
