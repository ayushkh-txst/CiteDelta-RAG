"""Ground truth and recall, with ties handled correctly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from citedelta.index.vector import Ids, Neighbor, Vectors

# Distances are float32 dot products; two mathematically-equal values can
# differ in the last bits. Without slack, a genuine tie reads as a miss.
TIE_EPSILON = 1e-5


@dataclass(frozen=True)
class GroundTruth:
    ids: NDArray[np.int64]  # (n_queries, k)
    distances: NDArray[np.float32]  # (n_queries, k)

    @property
    def k(self) -> int:
        return int(self.ids.shape[1])


def compute_ground_truth(
    corpus: Vectors, ids: Ids, queries: Vectors, k: int, *, batch: int = 64
) -> GroundTruth:
    """Exact k-NN for every query, by exhaustive scan.

    Batched because the full similarity matrix is (n_queries x n_corpus):
    at 500 x 38,211 that is 76 MB of float32 — survivable but pointless.
    64 queries at a time keeps it under 10 MB at the same speed.
    """
    out_ids = np.zeros((len(queries), k), dtype=np.int64)
    out_dist = np.zeros((len(queries), k), dtype=np.float32)

    for start in range(0, len(queries), batch):
        chunk = queries[start : start + batch]
        distances = 1.0 - (chunk @ corpus.T)  # (b, n)
        part = np.argpartition(distances, k - 1, axis=1)[:, :k]

        for row in range(len(chunk)):
            cand = part[row]
            # Same deterministic tie-break as the oracle: distance, then id.
            order = cand[np.lexsort((ids[cand], distances[row, cand]))]
            out_ids[start + row] = ids[order]
            out_dist[start + row] = distances[row, order]

    return GroundTruth(ids=out_ids, distances=out_dist)


def recall_with_ties(
    retrieved: list[Neighbor], truth_distances: NDArray[np.float32], k: int
) -> float:
    """Fraction of the top-k that is AS GOOD AS the true top-k.

    A result counts if its distance is within the k-th ground-truth distance.
    On a corpus with duplicate vectors this is the only correct definition:
    id-based matching penalizes an index for breaking an exact tie
    differently, which is not an error.
    """
    if k <= 0 or len(truth_distances) == 0:
        return 0.0
    threshold = float(truth_distances[min(k, len(truth_distances)) - 1]) + TIE_EPSILON
    return sum(1 for n in retrieved[:k] if n.distance <= threshold) / k


def recall_by_id(retrieved: list[Neighbor], truth_ids: NDArray[np.int64], k: int) -> float:
    """The naive definition. Reported ONLY so the gap to the tie-aware
    number is visible — that gap is a measure of corpus duplication, and
    hiding it would be hiding why the two numbers differ."""
    if k <= 0:
        return 0.0
    return len({n.id for n in retrieved[:k]} & set(truth_ids[:k].tolist())) / k
