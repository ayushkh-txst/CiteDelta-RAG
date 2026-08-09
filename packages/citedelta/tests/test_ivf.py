"""IVF behaviour that isn't shared: clustering, and the nprobe dial."""

from __future__ import annotations

import numpy as np
import pytest

from citedelta.index.brute import BruteForceIndex
from citedelta.index.ivf import IVFFlatIndex
from citedelta.index.kmeans import kmeans


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


def clustered(n_clusters: int, per_cluster: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = unit(rng.standard_normal((n_clusters, dim)).astype(np.float32))
    pts = np.repeat(centres, per_cluster, axis=0)
    pts = pts + 0.02 * rng.standard_normal(pts.shape).astype(np.float32)
    return unit(pts)


def test_kmeans_recovers_known_clusters() -> None:
    vectors = clustered(6, 60, 32)
    _, assignments = kmeans(vectors, 6, seed=0)
    # Each true group of 60 should land in one cell.
    for group in range(6):
        labels = assignments[group * 60 : (group + 1) * 60]
        assert len(set(labels.tolist())) == 1


def test_kmeans_handles_more_clusters_than_distinct_points() -> None:
    """Degenerate input must not produce NaN centroids."""
    vectors = np.repeat(unit(np.ones((1, 8), dtype=np.float32)), 5, axis=0)
    centroids, assignments = kmeans(vectors, 10, seed=0)
    assert np.isfinite(centroids).all()
    assert len(assignments) == 5


def test_no_empty_lists_after_build() -> None:
    ix = IVFFlatIndex(n_lists=16, seed=0)
    ix.build(np.arange(400, dtype=np.int64), clustered(16, 25, 24))
    assert ix.list_stats().empty == 0


def test_more_probes_never_reduce_recall() -> None:
    """The knob has to be monotonic, or it is not a knob."""
    vectors = clustered(20, 40, 32)
    ids = np.arange(800, dtype=np.int64)
    oracle = BruteForceIndex()
    oracle.build(ids, vectors)
    ix = IVFFlatIndex(n_lists=20, seed=0)
    ix.build(ids, vectors)

    queries = vectors[::40]
    scores = []
    for probe in (1, 2, 5, 20):
        total = 0.0
        for q in queries:
            truth = {h.id for h in oracle.search(q, 10)}
            got = {h.id for h in ix.search(q, 10, effort=probe)}
            total += len(truth & got) / 10
        scores.append(total / len(queries))

    assert scores == sorted(scores), f"recall not monotonic in nprobe: {scores}"
    assert scores[-1] == pytest.approx(1.0)  # nprobe == nlist is exact


def test_probing_every_cell_equals_brute_force() -> None:
    vectors = clustered(8, 30, 16)
    ids = np.arange(240, dtype=np.int64)
    oracle = BruteForceIndex()
    oracle.build(ids, vectors)
    ix = IVFFlatIndex(n_lists=8, seed=0)
    ix.build(ids, vectors)

    for q in vectors[:10]:
        assert [h.distance for h in ix.search(q, 10, effort=8)] == pytest.approx(
            [h.distance for h in oracle.search(q, 10)], abs=1e-5
        )


def test_default_n_lists_follows_sqrt_n() -> None:
    ix = IVFFlatIndex(seed=0)
    ix.build(np.arange(900, dtype=np.int64), clustered(30, 30, 16))
    assert ix.n_lists == 30  # sqrt(900)
