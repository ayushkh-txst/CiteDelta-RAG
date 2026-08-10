"""IVF-Flat: coarse quantizer + inverted lists + nprobe."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from citedelta.index.kmeans import kmeans
from citedelta.index.vector import (
    BoolMask,
    Ids,
    Neighbor,
    Vectors,
    compile_mask,
)


@dataclass(frozen=True, slots=True)
class ListStats:
    n_lists: int
    smallest: int
    largest: int
    mean: float
    empty: int

    @property
    def imbalance(self) -> float:
        """largest / mean. 1.0 is perfect; high means one cell dominates."""
        return self.largest / self.mean if self.mean else 0.0


class IVFFlatIndex:
    """Inverted file with flat (uncompressed) vectors.

    Storage is CSR-shaped rather than a dict of lists: vectors are PERMUTED
    into cluster order at build time, so each inverted list is a contiguous
    slice. Probing a cell is then one slice and one matmul over memory that
    is already sequential. A dict-of-arrays layout does the same arithmetic
    with scattered reads and is markedly slower for the same recall — this is
    a memory-layout win, not an algorithmic one.
    """

    def __init__(
        self,
        n_lists: int | None = None,
        *,
        n_probe: int | None = None,
        iterations: int = 25,
        seed: int = 0,
    ) -> None:
        self._requested_lists = n_lists
        self._default_probe = n_probe
        self._iterations = iterations
        self._seed = seed

        self._centroids: Vectors = np.zeros((0, 0), dtype=np.float32)
        self._vectors: Vectors = np.zeros((0, 0), dtype=np.float32)
        self._ids: Ids = np.zeros(0, dtype=np.int64)
        self._offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self._last_probes = 0

    @property
    def name(self) -> str:
        return "ivf-flat"

    @property
    def size(self) -> int:
        return int(self._ids.shape[0])

    @property
    def dimensions(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.size else 0

    @property
    def n_lists(self) -> int:
        return int(self._centroids.shape[0])

    def _choose_n_lists(self, n: int) -> int:
        """sqrt(N) is the standard heuristic, and it is not arbitrary.

        Query cost is roughly `nlist` (centroid comparisons) plus `N/nlist`
        (one cell's worth of vectors). Minimising nlist + N/nlist gives
        nlist = sqrt(N) — the point where the two halves balance.
        """
        return self._requested_lists or max(1, min(n, int(math.sqrt(n))))

    def build(self, ids: Ids, vectors: Vectors) -> None:
        if len(ids) != len(vectors):
            msg = f"{len(ids)} ids but {len(vectors)} vectors"
            raise ValueError(msg)

        ids = np.ascontiguousarray(ids, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(ids) == 0:
            self._ids, self._vectors = ids, vectors
            return

        n_lists = self._choose_n_lists(len(ids))
        self._centroids, assignments = kmeans(
            vectors, n_lists, iterations=self._iterations, seed=self._seed
        )

        # Permute into cluster order so each list is one contiguous slice.
        # 'stable' keeps ids ascending inside a cell, which makes tie-breaking
        # reproducible without a second sort key at query time.
        order = np.argsort(assignments, kind="stable")
        self._vectors = np.ascontiguousarray(vectors[order])
        self._ids = np.ascontiguousarray(ids[order])
        self._offsets = np.searchsorted(
            assignments[order], np.arange(self.n_lists + 1), side="left"
        ).astype(np.int64)

    def list_stats(self) -> ListStats:
        sizes = np.diff(self._offsets)
        return ListStats(
            n_lists=self.n_lists,
            smallest=int(sizes.min()) if len(sizes) else 0,
            largest=int(sizes.max()) if len(sizes) else 0,
            mean=float(sizes.mean()) if len(sizes) else 0.0,
            empty=int((sizes == 0).sum()),
        )

    def compile_filter(self, admissible_ids: Collection[int]) -> BoolMask:
        return compile_mask(self._ids, admissible_ids)

    @property
    def last_probes_used(self) -> int:
        """Cells probed by the most recent search.

        Diagnostic only, and deliberately NOT thread-safe — it is mutable state
        on a read-only index. The benchmark is single-threaded, so it is
        correct there; a concurrent server would need this returned from
        `search` instead of stashed. Flagging it rather than pretending the
        index is immutable.
        """
        return self._last_probes

    def search(
        self,
        query: Vectors,
        k: int,
        *,
        effort: int | None = None,
        admissible: BoolMask | None = None,
    ) -> list[Neighbor]:
        """`effort` is nprobe. Under a filter it becomes a MINIMUM, not a fixed
        budget: probing continues until k admissible candidates are found."""
        if self.size == 0 or k <= 0:
            return []

        k = min(k, self.size)
        n_probe = effort or self._default_probe or max(1, self.n_lists // 16)
        n_probe = max(1, min(n_probe, self.n_lists))

        # Step 1: rank the cells. nlist dot products, cheap.
        centroid_distance = 1.0 - (self._centroids @ query)

        if admissible is None:
            probes = np.argpartition(centroid_distance, n_probe - 1)[:n_probe]
            spans = [np.arange(self._offsets[c], self._offsets[c + 1]) for c in probes]
            candidates = np.concatenate(spans) if spans else np.zeros(0, dtype=np.int64)
            self._last_probes = n_probe
        else:
            candidates = self._probe_until_k(query, k, n_probe, centroid_distance, admissible)

        if candidates.size == 0:
            return []

        # Step 3: exact scan within the candidates.
        distances = 1.0 - (self._vectors[candidates] @ query)
        take = min(k, len(candidates))
        top = np.argpartition(distances, take - 1)[:take]
        order = top[np.lexsort((self._ids[candidates[top]], distances[top]))]

        return [
            Neighbor(id=int(self._ids[candidates[i]]), distance=float(distances[i])) for i in order
        ]

    def _probe_until_k(
        self,
        query: Vectors,
        k: int,
        min_probes: int,
        centroid_distance: np.ndarray,
        admissible: BoolMask,
    ) -> np.ndarray:
        """Walk cells nearest-first, keeping only admissible rows, until k found.

        A FULL argsort of the centroids rather than argpartition: nlist is ~194,
        so sorting costs microseconds, and we genuinely may need the complete
        ordering — under a very selective filter this loop can walk every cell.
        argpartition would only give the nearest `nprobe`, and then we would have
        no defined order to continue in.

        Correctness note: this returns the k nearest admissible rows *within the
        probed cells*, which is still approximate. A closer admissible row can
        sit in a cell we stopped before reaching. It stops being approximate
        only when every cell is probed — the same guarantee unfiltered IVF has,
        preserved rather than improved.
        """
        cell_order = np.argsort(centroid_distance)
        collected: list[np.ndarray] = []
        found = 0
        probed = 0

        for cell in cell_order:
            lo = int(self._offsets[cell])
            hi = int(self._offsets[cell + 1])
            probed += 1
            if hi > lo:
                span = np.arange(lo, hi)
                span = span[admissible[span]]
                if span.size:
                    collected.append(span)
                    found += int(span.size)
            if probed >= min_probes and found >= k:
                break

        self._last_probes = probed
        return np.concatenate(collected) if collected else np.zeros(0, dtype=np.int64)

    def memory_bytes(self) -> int:
        return int(
            self._vectors.nbytes + self._ids.nbytes + self._centroids.nbytes + self._offsets.nbytes
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        with tmp.open("wb") as fh:
            np.savez(
                fh,
                ids=self._ids,
                vectors=self._vectors,
                centroids=self._centroids,
                offsets=self._offsets,
            )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        data = np.load(path, allow_pickle=False)
        index = cls()
        index._ids = data["ids"]
        index._vectors = data["vectors"]
        index._centroids = data["centroids"]
        index._offsets = data["offsets"]
        return index
