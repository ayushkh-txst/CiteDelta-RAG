"""asyncpg connection pooling with an explicit, stated size."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Self

import asyncpg


async def _init_connection(conn: asyncpg.Connection[Any]) -> None:
    """Make jsonb round-trip as dict instead of str."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


class Database:
    """A pool with a size you chose on purpose."""

    def __init__(self, dsn: str, *, min_size: int = 2, max_size: int = 10) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool[Any] | None = None

    @property
    def pool(self) -> asyncpg.Pool[Any]:
        if self._pool is None:
            msg = "Database.connect() has not been awaited"
            raise RuntimeError(msg)
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            init=_init_connection,
            command_timeout=60,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection[Any]]:
        async with self.pool.acquire() as conn:
            yield conn

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @classmethod
    @asynccontextmanager
    async def open(cls, dsn: str, **kw: int) -> AsyncIterator[Database]:
        db = cls(dsn, **kw)
        await db.connect()
        try:
            yield db
        finally:
            await db.close()
