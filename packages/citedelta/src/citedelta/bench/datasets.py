"""Benchmark datasets. Held-out queries, so nothing self-matches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from citedelta.index.vector import Ids, Vectors

N_QUERIES = 500


@dataclass(frozen=True)
class Dataset:
    """A train/test split, as every ANN benchmark uses.

    Queries are HELD OUT of the index rather than sampled from it. Sampling
    from the corpus means every query's nearest neighbour is itself at
    distance 0, and then every index has to special-case self-exclusion —
    a whole class of off-by-one bugs, in the metric, avoided by construction.
    """

    name: str
    ids: Ids
    vectors: Vectors
    queries: Vectors
    note: str = ""

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1])


def _split(name: str, ids: Ids, vectors: Vectors, *, seed: int = 0, note: str = "") -> Dataset:
    rng = np.random.default_rng(seed)
    n_queries = min(N_QUERIES, max(1, len(ids) // 10))
    held_out = rng.choice(len(ids), n_queries, replace=False)
    mask = np.ones(len(ids), dtype=bool)
    mask[held_out] = False
    return Dataset(
        name=name,
        ids=ids[mask],
        vectors=vectors[mask],
        queries=vectors[held_out],
        note=note,
    )


def cfr_full(ids: Ids, vectors: Vectors, *, seed: int = 0) -> Dataset:
    """The production corpus, duplicates and all."""
    return _split("cfr-full", ids, vectors, seed=seed, note="real corpus, as indexed in production")


def cfr_dedup(ids: Ids, vectors: Vectors, *, seed: int = 0) -> Dataset:
    """Distinct vectors only.

    Duplicates inflate the corpus without adding geometric variety, which
    makes the search problem easier than the row count suggests. Reporting
    both sizes separates 'my index is good' from 'my corpus is repetitive'.
    """
    _, first = np.unique(vectors, axis=0, return_index=True)
    keep = np.sort(first)
    return _split(
        "cfr-dedup", ids[keep], vectors[keep], seed=seed, note="one vector per distinct text"
    )


def random_hard(n: int = 20_000, dim: int = 384, *, seed: int = 0) -> Dataset:
    """Uniform random unit vectors — the ANN worst case, on purpose.

    In high dimensions random points concentrate: every pairwise distance
    converges on the same value, so 'nearest neighbour' is barely meaningful
    and a graph index has almost no gradient to follow.

    It is in the suite for exactly one reason: the real corpus has intrinsic
    dimension ~1.1 and returns recall 1.000 at every setting, so it cannot
    show a recall/speed tradeoff. This dataset can. A benchmark that only
    ever reports 1.0 has not demonstrated that the knob works.

    It is NOT representative of the product data, and the report must say so.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(n, dtype=np.int64)
    return _split(
        "random-hard", ids, vectors, seed=seed, note="uniform random unit vectors: ANN worst case"
    )
