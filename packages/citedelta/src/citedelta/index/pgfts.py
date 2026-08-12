"""Postgres full-text search behind the LexicalIndex shape.

BM25 vs ts_rank_cd is not apples to apples and the write-up must say so:
Postgres ranks by term-frequency and proximity, without BM25's document
length normalisation or its saturating tf. Expect different orderings even
where both are 'correct'. The comparison that matters is recall against the
same ground truth and latency at the same k, not score agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg


@dataclass(frozen=True, slots=True)
class FtsHit:
    chunk_id: int
    score: float


class PgFullTextIndex:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)

    def search(self, query: str, k: int, *, admissible: list[int] | None = None) -> list[FtsHit]:
        with self._conn.cursor() as cur:
            if admissible is None:
                cur.execute(
                    """SELECT id, ts_rank_cd(ts, plainto_tsquery('english', %s)) AS r
                       FROM chunks WHERE ts @@ plainto_tsquery('english', %s)
                       ORDER BY r DESC LIMIT %s""",
                    (query, query, k),
                )
            else:
                cur.execute(
                    """SELECT id, ts_rank_cd(ts, plainto_tsquery('english', %s)) AS r
                       FROM chunks
                       WHERE ts @@ plainto_tsquery('english', %s) AND id = ANY(%s)
                       ORDER BY r DESC LIMIT %s""",
                    (query, query, admissible, k),
                )
            return [FtsHit(int(r[0]), float(r[1])) for r in cur.fetchall()]
