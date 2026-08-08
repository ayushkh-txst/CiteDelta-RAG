"""Shared fixtures. Migrations run once per session against a real Postgres."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from substrate.db import Database

ROOT = Path(__file__).parent
DSN = os.environ.get("DATABASE_URL", "postgresql://citedelta:citedelta@localhost:5434/citedelta")

TABLES = ("chunks", "section_versions", "documents", "sources")


@pytest.fixture(scope="session", autouse=True)
def _migrated() -> Iterator[None]:
    """Bring the schema to head before anything runs.

    Deliberately SYNC: alembic's async env.py calls asyncio.run(), which
    explodes if there is already a running event loop. Sync session scope
    means this completes before pytest-asyncio starts one.
    """
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    command.upgrade(cfg, "head")
    yield


@pytest.fixture
def dsn() -> str:
    return DSN


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    async with Database.open(DSN, min_size=1, max_size=5) as database:
        yield database


@pytest.fixture
async def clean_db(db: Database) -> AsyncIterator[Database]:
    """Empty corpus tables. Tests must not depend on each other's leftovers."""
    async with db.acquire() as conn:
        await conn.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    yield db
