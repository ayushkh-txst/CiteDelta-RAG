"""Exact k-NN by exhaustive scan. Slow, obviously correct, the oracle."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Self

import numpy as np

from citedelta.index.vector import (
    BoolMask,
    Ids,
    Neighbor,
    Vectors,
    compile_mask,
    cosine_distance,
)


class BruteForceIndex:
    """Every number in `docs/design/03-benchmarks.md` is measured against this."""

    def __init__(self) -> None:
        self._ids: Ids = np.zeros(0, dtype=np.int64)
        self._vectors: Vectors = np.zeros((0, 0), dtype=np.float32)

    @property
    def name(self) -> str:
        return "brute-force"

    @property
    def size(self) -> int:
        return int(self._ids.shape[0])

    @property
    def dimensions(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.size else 0

    def build(self, ids: Ids, vectors: Vectors) -> None:
        if len(ids) != len(vectors):
            msg = f"{len(ids)} ids but {len(vectors)} vectors"
            raise ValueError(msg)
        # C-contiguous float32 so the matmul hits the fast BLAS path. A
        # non-contiguous view here silently costs a copy per query.
        self._ids = np.ascontiguousarray(ids, dtype=np.int64)
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def compile_filter(self, admissible_ids: Collection[int]) -> BoolMask:
        return compile_mask(self._ids, admissible_ids)

    def search(
        self,
        query: Vectors,
        k: int,
        *,
        effort: int | None = None,
        admissible: BoolMask | None = None,
    ) -> list[Neighbor]:
        """Exact top-k, optionally restricted to an admissible subset.

        With a filter this is the FILTERED ORACLE — the exact answer to
        'what are the k nearest admissible neighbours', and therefore what
        every approximate filtered index is scored against.

        Implemented by setting inadmissible distances to +inf rather than by
        slicing the array. Slicing would allocate a copy of the admissible
        rows on every query (megabytes, per query); masking the distance
        vector touches N floats once and keeps row indices aligned with
        `self._ids`, so no index translation is needed afterwards.
        """
        if self.size == 0 or k <= 0:
            return []

        distances = cosine_distance(self._vectors, query)

        if admissible is not None:
            available = int(np.count_nonzero(admissible))
            if available == 0:
                return []
            k = min(k, available)
            distances = np.where(admissible, distances, np.inf).astype(np.float32)
        else:
            k = min(k, self.size)

        # argpartition is O(N) and only guarantees the k smallest are in the
        # first k slots, unordered. Sort only the k, not all N.
        candidates = np.argpartition(distances, k - 1)[:k]

        # Deterministic tie-breaking: the corpus holds exact duplicate texts
        # across versions, so identical vectors produce identical distances.
        # Without a stable secondary key the oracle would return a different
        # winner run to run. lexsort applies the LAST key first, so this is
        # (distance, then id).
        order = candidates[np.lexsort((self._ids[candidates], distances[candidates]))]

        return [Neighbor(id=int(self._ids[i]), distance=float(distances[i])) for i in order]

    def memory_bytes(self) -> int:
        return int(self._vectors.nbytes + self._ids.nbytes)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        with tmp.open("wb") as fh:
            np.savez(fh, ids=self._ids, vectors=self._vectors)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        data = np.load(path, allow_pickle=False)
        index = cls()
        index.build(data["ids"], data["vectors"])
        return index
