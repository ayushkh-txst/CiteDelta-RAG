"""The vector-index port. Three implementations, one conformance suite."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]
Ids = NDArray[np.int64]
BoolMask = NDArray[np.bool_]


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

    def compile_filter(self, admissible_ids: Collection[int]) -> BoolMask:
        """Compile an admissible id set into this index's internal ordering."""
        ...

    def search(
        self,
        query: Vectors,
        k: int,
        *,
        effort: int | None = None,
        admissible: BoolMask | None = None,
    ) -> list[Neighbor]: ...

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


def compile_mask(index_ids: Ids, admissible_ids: Collection[int]) -> BoolMask:
    """Admissible ids → boolean mask in an index's internal row order.

    One line, and it works identically for all three indexes — because every
    one of them keeps `self._ids` aligned to its own internal layout. IVF
    permutes its rows into cluster order and permutes `_ids` with them, so
    `_ids[i]` is always the external id of internal row i. That invariant is
    what makes filtering index-agnostic; break it in a future index and
    filtering breaks silently, with plausible-looking wrong results.

    np.isin rather than a Python set comprehension: at 38,211 x 728 this is a
    sort-and-search in C (about a millisecond) instead of 38,211 interpreter
    round-trips.
    """
    wanted = np.fromiter(admissible_ids, dtype=np.int64, count=len(admissible_ids))
    return np.isin(index_ids, wanted)
