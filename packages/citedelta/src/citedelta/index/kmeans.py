"""Spherical k-means for unit-normalized vectors."""

from __future__ import annotations

import numpy as np
import structlog

from citedelta.index.vector import Vectors

log = structlog.get_logger(__name__)


def kmeans_plusplus_init(vectors: Vectors, k: int, rng: np.random.Generator) -> Vectors:
    """Seed centroids far apart, rather than uniformly at random.

    Plain random init routinely drops two seeds into the same dense region,
    which wastes a cell and leaves a real cluster unrepresented — and IVF
    pays for that forever, because a badly-placed boundary is a permanently
    unreachable neighbour.

    k-means++ picks each new seed with probability proportional to its
    squared distance from the nearest seed already chosen: far-away points
    are likely, near-duplicates are not. Costs k passes over the corpus and
    buys a materially better partition.
    """
    n = len(vectors)
    centroids = np.empty((k, vectors.shape[1]), dtype=np.float32)
    centroids[0] = vectors[rng.integers(n)]

    closest = 1.0 - (vectors @ centroids[0])
    for i in range(1, k):
        weights = np.maximum(closest, 0.0) ** 2
        total = float(weights.sum())
        # Degenerate corpus (all points identical, or fewer distinct points
        # than k): fall back to uniform choice rather than dividing by zero.
        idx = rng.integers(n) if total <= 0 else int(rng.choice(n, p=weights / total))
        centroids[i] = vectors[idx]
        closest = np.minimum(closest, 1.0 - (vectors @ centroids[i]))

    return centroids


def assign(vectors: Vectors, centroids: Vectors, *, batch: int = 8192) -> np.ndarray:
    """Nearest centroid for every vector, one matmul per batch.

    Batched to bound the (batch x nlist) similarity matrix. At 38k x 256 the
    whole thing is small; the same code has to survive a corpus an order of
    magnitude larger.
    """
    out = np.empty(len(vectors), dtype=np.int32)
    for start in range(0, len(vectors), batch):
        sims = vectors[start : start + batch] @ centroids.T
        out[start : start + batch] = np.argmax(sims, axis=1)
    return out


def kmeans(
    vectors: Vectors, k: int, *, iterations: int = 25, seed: int = 0, tol: float = 1e-4
) -> tuple[Vectors, np.ndarray]:
    """Lloyd's algorithm, on the unit sphere.

    'Spherical' because the vectors are L2-normalized: after averaging the
    members of a cell we RE-NORMALIZE, which keeps every centroid on the same
    sphere as the data. Skip that and centroids drift toward the origin,
    where the dot product no longer means cosine similarity and assignment
    quietly degrades.
    """
    k = min(k, len(vectors))
    rng = np.random.default_rng(seed)
    centroids = kmeans_plusplus_init(vectors, k, rng)
    assignments = np.zeros(len(vectors), dtype=np.int32)

    for iteration in range(iterations):
        new_assignments = assign(vectors, centroids)
        changed = float(np.mean(new_assignments != assignments))
        assignments = new_assignments

        for cell in range(k):
            members = vectors[assignments == cell]
            if len(members) == 0:
                # An empty cell is wasted capacity AND a latent crash (mean of
                # nothing is NaN, which poisons every later assignment). Re-seed
                # it on the point currently worst served by its own centroid.
                worst = int(np.argmin(np.max(vectors @ centroids.T, axis=1)))
                centroids[cell] = vectors[worst]
                continue
            centre = members.mean(axis=0)
            norm = float(np.linalg.norm(centre))
            centroids[cell] = centre / norm if norm > 1e-12 else vectors[rng.integers(len(vectors))]

        if changed < tol:
            log.info("kmeans.converged", iteration=iteration + 1, changed=round(changed, 5))
            break

    return centroids, assign(vectors, centroids)
