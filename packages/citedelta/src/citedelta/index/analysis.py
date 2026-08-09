"""Corpus geometry. What you measure here explains every recall number."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from citedelta.index.vector import Vectors


@dataclass(frozen=True, slots=True)
class CorpusGeometry:
    n_vectors: int
    n_distinct: int
    ambient_dim: int
    intrinsic_dim: float
    mean_nn_distance: float

    @property
    def duplicate_ratio(self) -> float:
        return self.n_vectors / max(self.n_distinct, 1)


def intrinsic_dimension(vectors: Vectors, *, sample: int = 2000, seed: int = 0) -> float:
    """Two-NN maximum-likelihood estimator (Facco et al., 2017).

    For each point take its two nearest neighbours at distances r1 <= r2 and
    form mu = r2/r1. Under a locally-uniform density on a d-dimensional
    manifold, mu follows a Pareto distribution whose shape parameter IS d.
    The MLE is then simply  d = n / sum(log mu).

    Exact duplicates must be removed first: they give r1 = 0, so mu is
    infinite and the estimator divides by zero. The corpus is full of them
    (the same paragraph text recurs across versions), so this is a real
    failure mode, not a theoretical one.
    """
    unique = np.unique(vectors, axis=0)
    rng = np.random.default_rng(seed)
    m = min(sample, len(unique))
    idx = rng.choice(len(unique), m, replace=False)

    sims = unique[idx] @ unique.T
    sims[np.arange(m), idx] = -np.inf  # exclude self-match
    # True Euclidean distance: for unit vectors ||a-b||^2 = 2(1 - cos).
    # Using raw 1 - cos here halves the MLE, since it scales as distance^2.
    chord = np.sqrt(np.maximum(2.0 * (1.0 - sims), 0.0))
    nearest_two = np.sort(chord, axis=1)[:, :2]

    r1 = np.maximum(nearest_two[:, 0], 1e-12)
    mu = nearest_two[:, 1] / r1
    log_mu = np.log(np.maximum(mu, 1.0 + 1e-12))
    return float(m / np.sum(log_mu))


def describe(vectors: Vectors, *, sample: int = 2000, seed: int = 0) -> CorpusGeometry:
    unique = np.unique(vectors, axis=0)
    rng = np.random.default_rng(seed)
    m = min(sample, len(unique))
    idx = rng.choice(len(unique), m, replace=False)
    sims = unique[idx] @ unique.T
    sims[np.arange(m), idx] = -np.inf
    nn = np.sort(1.0 - sims, axis=1)[:, 0]

    return CorpusGeometry(
        n_vectors=len(vectors),
        n_distinct=len(unique),
        ambient_dim=int(vectors.shape[1]),
        intrinsic_dim=intrinsic_dimension(vectors, sample=sample, seed=seed),
        mean_nn_distance=float(np.mean(nn)),
    )
