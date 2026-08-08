"""Read chunks out of Postgres and build the lexical index."""

from __future__ import annotations

from pathlib import Path

import structlog

from citedelta.config import get_settings
from citedelta.index.lexical import BuildStats, build_index
from substrate.db import Database

log = structlog.get_logger(__name__)

LEXICAL_INDEX_FILENAME = "lexical.idx"


async def build_lexical_index(path: Path | None = None) -> BuildStats:
    """Index EVERY chunk version, not just what is in force today.

    A temporal query has to be able to reach historical text, so the index
    must contain it. Restricting the index to 'current' would make 'what did
    the rule say in 2019?' unanswerable — which is the one question this
    project exists to answer.

    The index therefore holds several versions of near-identical text, and a
    naive search returns all of them. Filtering by validity is planned as an
    indexed pushdown.
    """
    settings = get_settings()
    target = path or (settings.index_dir / LEXICAL_INDEX_FILENAME)

    async with Database.open(settings.database_url) as db, db.acquire() as conn:
        rows = await conn.fetch("SELECT id, text FROM chunks ORDER BY id")

    stats = build_index(((int(r["id"]), str(r["text"])) for r in rows), target)
    log.info("index.persisted", path=str(target))
    return stats
