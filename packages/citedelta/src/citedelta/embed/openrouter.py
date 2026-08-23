"""OpenRouter-hosted embeddings, truncated to a smaller Matryoshka prefix.

Synchronous by design, matching `EmbeddingProvider` — `LocalEmbeddings` was
synchronous because ONNX inference is CPU-bound and fast; this is a network
call instead, so callers on the request path must run it off the event loop
(see `anyio.to_thread.run_sync` at the two query-time call sites in
`api/app.py`). Keeping the port sync rather than infecting it with async
keeps `AnswerService`/the indexes untouched.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
import structlog

from citedelta.config import Settings
from citedelta.embed.base import Vectors
from substrate.resilience import full_jitter_delay

log = structlog.get_logger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/text-embedding-3-small"
DEFAULT_DIMENSIONS = 512
"""Matryoshka-truncated from the model's native 1536. 512 keeps ~38k chunks'
worth of vectors around 78MB — comfortable inside a free 500MB Postgres —
where the full 1536 would be ~233MB. text-embedding-3-small is trained so
that a leading prefix of the vector is independently meaningful, which is
what makes truncate-then-renormalize a reasonable trade rather than noise."""

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    """A batch that failed after exhausting retries."""


def default_provider(settings: Settings) -> OpenRouterEmbeddings:
    """Build the provider a plain `citedelta.config.Settings` describes.

    A small function rather than repeating this construction at every call
    site (`embed/corpus.py`, `api/state.py`) that wants "whatever embedding
    model the deployment is currently configured for."
    """
    return OpenRouterEmbeddings(
        api_key=settings.openrouter_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )


class OpenRouterEmbeddings:
    """Text in, unit-normalized float32 vectors out, via OpenRouter.

    Implements `EmbeddingProvider` (see embed/base.py) without inheriting
    from it — the protocol is structural.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        max_attempts: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
            timeout=60.0,
        )
        self._model = model
        self._dimensions = dimensions
        self._max_attempts = max_attempts

    @property
    def model_id(self) -> str:
        # Distinct from the bare model name on purpose: the `embeddings`
        # table is keyed by (model_id, content_sha256), and this truncation
        # produces a different vector space than the full 1536-dim model —
        # a shared model_id would let the two silently collide.
        return f"{self._model}@{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> Vectors:
        if not texts:
            return np.zeros((0, self._dimensions), dtype=np.float32)

        out = np.empty((len(texts), self._dimensions), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            out[start : start + len(batch)] = self._embed_batch(batch)
        return out

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._client.post("/embeddings", json={"model": self._model, "input": batch})
            except httpx.TransportError as exc:
                last = exc
                if attempt < self._max_attempts:
                    self._backoff(attempt, exc)
                continue

            if raw.status_code in _RETRYABLE_STATUS:
                last = httpx.HTTPStatusError(
                    f"{raw.status_code} from OpenRouter", request=raw.request, response=raw
                )
                if attempt < self._max_attempts:
                    self._backoff(attempt, last)
                continue

            if raw.status_code >= 400:
                raise EmbeddingError(f"HTTP {raw.status_code} embedding a batch: {raw.text[:200]}")

            return self._to_vectors(raw.json(), n=len(batch))

        raise EmbeddingError(f"{self._max_attempts} attempts failed embedding a batch") from last

    def _backoff(self, attempt: int, exc: Exception) -> None:
        delay = full_jitter_delay(attempt)
        log.warning(
            "embed.retry", attempt=attempt, delay_s=round(delay, 3), error=type(exc).__name__
        )
        time.sleep(delay)

    def _to_vectors(self, raw: dict[str, Any], *, n: int) -> np.ndarray:
        # Row order is not guaranteed by the response order — sort by the
        # `index` the API returns, same discipline the protocol demands of
        # every implementation.
        rows = sorted(raw["data"], key=lambda d: d["index"])
        full = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        if full.shape[0] != n:
            raise EmbeddingError(f"expected {n} embeddings, got {full.shape[0]}")

        truncated = full[:, : self._dimensions]
        # Truncating a Matryoshka embedding leaves it no longer unit-length —
        # renormalizing here is mandatory, not the no-op insurance it is in
        # LocalEmbeddings.
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        np.divide(truncated, np.maximum(norms, 1e-12), out=truncated)
        return truncated
