"""HTTP access to the eCFR versioner API."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
import structlog

from citedelta.ecfr.models import VersionRecord

log = structlog.get_logger(__name__)

BASE_URL = "https://www.ecfr.gov/api/versioner/v1"


class ECFRClient:
    """Politely rate-limited, disk-cached client for the public eCFR API.

    min_interval enforces a floor between requests so bulk fetches don't get
    the IP blocked. A past snapshot never changes, so cached XML is safe to
    reuse forever.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        min_interval: float = 0.34,  # ~3 req/s
        timeout: float = 90.0,
    ) -> None:
        self._cache_dir = cache_dir
        self._min_interval = min_interval
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"User-Agent": "CiteDelta/0.1 (contact via GitHub)"},
            follow_redirects=True,
        )
        self._last_request = 0.0
        self._gate = asyncio.Lock()

    async def _throttle(self) -> None:
        async with self._gate:
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

    async def versions(self, title: int, part: str) -> list[VersionRecord]:
        """Every tracked version of every section in a part. Not disk-cached —
        it is small and it is the one thing that legitimately changes."""
        await self._throttle()
        resp = await self._client.get(f"/versions/title-{title}.json", params={"part": part})
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        return [VersionRecord.model_validate(r) for r in payload["content_versions"]]

    async def snapshot_xml(self, title: int, part: str, on: date) -> bytes:
        """The full text of a part as it stood on a given date.

        Cached to disk forever — a past snapshot is immutable, so a cache hit
        is always correct and re-runs cost zero requests.
        """
        path = self._cache_path(title, part, on)
        if path.exists():
            log.debug("snapshot.cache_hit", date=str(on), bytes=path.stat().st_size)
            return path.read_bytes()

        await self._throttle()
        log.info("snapshot.fetch", title=title, part=part, date=str(on))
        resp = await self._client.get(
            f"/full/{on.isoformat()}/title-{title}.xml", params={"part": part}
        )
        resp.raise_for_status()
        body = resp.content
        self._write_atomic(path, body)
        return body

    def _cache_path(self, title: int, part: str, on: date) -> Path:
        return self._cache_dir / f"title-{title}" / f"part-{part}" / f"{on.isoformat()}.xml"

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        """Write to a temp file, fsync, then rename so readers never see a
        half-written file. os.replace() is atomic on POSIX."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
