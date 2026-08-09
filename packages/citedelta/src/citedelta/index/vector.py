"""The vector-index port. Three implementations, one conformance suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]
Ids = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Neighbor:
    """A search result. LOWER distance is better — always, in every index.

    Naming this `distance` rather than `score` is deliberate. The lexical
    index returns higher-is-better BM25 scores; these return lower-is-better
    distances. Mixing the two conventions silently inverts a ranking, and
    an inverted ranking still looks like a plausible list of results. The
    field name is the guardrail.
    """

    id: int
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity, for display. Vectors are unit-normalized."""
        return 1.0 - self.distance


@runtime_checkable
class VectorIndex(Protocol):
    """Everything an index must do to be benchmarked by the same harness.

    On the `effort` parameter: each index has exactly one knob trading
    accuracy for speed — `nprobe` for IVF, `ef_search` for HNSW — so the
    protocol names the CONCEPT and each implementation interprets it in its
    own units. That is what lets the harness sweep every index with one loop
    instead of a special case per type.

    None means 'use this index's default'. Brute force ignores it entirely,
    because exact search has no accuracy knob — which is itself the point.
    """

    @property
    def name(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def dimensions(self) -> int: ...

    def build(self, ids: Ids, vectors: Vectors) -> None: ...

    def search(self, query: Vectors, k: int, *, effort: int | None = None) -> list[Neighbor]: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> Self: ...

    def memory_bytes(self) -> int:
        """Resident size, computed analytically. Reported in the benchmark."""
        ...


def cosine_distance(vectors: Vectors, query: Vectors) -> NDArray[np.float32]:
    """1 - cosine similarity, for UNIT-NORMALIZED vectors only.

    Because ||a|| = ||b|| = 1, cosine(a,b) = a·b, so the whole distance
    computation over a corpus is a single matrix-vector product. That
    identity is why the EmbeddingProvider protocol promises normalized
    output — it turns similarity into BLAS.
    """
    return 1.0 - (vectors @ query)
