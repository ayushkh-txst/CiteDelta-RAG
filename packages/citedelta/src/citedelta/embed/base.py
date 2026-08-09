"""The embedding port. Domain code depends on this, never on a vendor SDK."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text in, unit-normalized float32 vectors out.

    Two guarantees every implementation must honour, because the indexes
    downstream assume both:

    1. Output is L2-NORMALIZED. Cosine similarity then equals the dot
       product, which turns every similarity computation in every index into
       a matmul. If an implementation cannot guarantee it, it normalizes.
    2. Output row order matches input order. Obvious, easy to break the
       moment anyone adds concurrency.
    """

    @property
    def model_id(self) -> str:
        """Stable identifier. Goes into the cache key — see cache.py."""
        ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> Vectors: ...
