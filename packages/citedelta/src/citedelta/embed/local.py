"""Local ONNX embeddings. No API key, no network at query time, deterministic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

from citedelta.embed.base import Vectors

log = structlog.get_logger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSIONS = 384


class LocalEmbeddings:
    """fastembed + bge-small-en-v1.5, 384 dimensions.

    Why local rather than a hosted API (ADR-0005):

    * DETERMINISM. Today's whole deliverable is a recall benchmark. A hosted
      model can be re-versioned behind a stable name, and then Tuesday's
      recall@10 isn't comparable with Monday's. A pinned local model makes
      the numbers reproducible, which is the only thing that makes them
      worth publishing.
    * $0, and no key in the query path.
    * Quality is genuinely lower than a frontier embedding model. That's a
      real cost, quantified in the service load test rather than hand-waved.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, *, threads: int | None = None) -> None:
        from fastembed import TextEmbedding  # imported lazily: ~2 s and ~200 MB

        self._model_id = model_id
        self._model: Any = TextEmbedding(model_id, threads=threads)
        log.info("embeddings.model_loaded", model=model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return DEFAULT_DIMENSIONS

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        vecs = np.asarray(
            list(self._model.embed(list(texts), batch_size=batch_size)),
            dtype=np.float32,
        )

        # bge-small already returns unit vectors — I measured ||v|| = 1.000000.
        # Re-normalizing is a no-op here, and it is the cheapest possible
        # insurance against a future provider that doesn't. The protocol
        # promises normalized output; the protocol has to be true for every
        # implementation, not just the one that happens to be plugged in.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        np.divide(vecs, np.maximum(norms, 1e-12), out=vecs)
        return vecs
