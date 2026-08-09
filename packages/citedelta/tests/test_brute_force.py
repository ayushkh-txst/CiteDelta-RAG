"""Exact search, and the tie-determinism the oracle depends on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from citedelta.index.analysis import intrinsic_dimension
from citedelta.index.brute import BruteForceIndex
from citedelta.index.vector import Neighbor, VectorIndex


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.linalg.norm(v, axis=-1, keepdims=True))


@pytest.fixture
def index() -> BruteForceIndex:
    vectors = unit(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],  # exact duplicate of id 10 — ties on purpose
            ],
            dtype=np.float32,
        )
    )
    ix = BruteForceIndex()
    ix.build(np.array([10, 20, 30, 40, 50], dtype=np.int64), vectors)
    return ix


def test_satisfies_the_protocol(index: BruteForceIndex) -> None:
    assert isinstance(index, VectorIndex)


def test_finds_the_exact_match_first(index: BruteForceIndex) -> None:
    hits = index.search(unit(np.array([1.0, 0.0, 0.0])), k=1)
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    assert hits[0].id in (10, 50)


def test_results_are_sorted_by_distance(index: BruteForceIndex) -> None:
    hits = index.search(unit(np.array([1.0, 0.05, 0.0])), k=5)
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_ties_break_deterministically(index: BruteForceIndex) -> None:
    """Ids 10 and 50 are identical vectors. The oracle must not waver.

    Recall is measured against this index's output. If a tie resolved
    differently between runs, recall would move for reasons that have nothing
    to do with the index being measured.
    """
    q = unit(np.array([1.0, 0.0, 0.0]))
    first = [h.id for h in index.search(q, k=5)]
    for _ in range(20):
        assert [h.id for h in index.search(q, k=5)] == first
    assert first[:2] == [10, 50]  # equal distance -> lower id first


def test_k_larger_than_corpus_is_clamped(index: BruteForceIndex) -> None:
    assert len(index.search(unit(np.array([1.0, 0.0, 0.0])), k=99)) == 5


def test_empty_index_returns_nothing() -> None:
    assert BruteForceIndex().search(np.zeros(3, dtype=np.float32), k=5) == []


def test_never_returns_an_id_it_was_not_given(index: BruteForceIndex) -> None:
    got = {h.id for h in index.search(unit(np.array([0.3, 0.4, 0.5])), k=5)}
    assert got <= {10, 20, 30, 40, 50}


def test_mismatched_ids_and_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="ids but"):
        BruteForceIndex().build(
            np.array([1, 2], dtype=np.int64), unit(np.ones((3, 3), dtype=np.float32))
        )


def test_save_load_round_trip(index: BruteForceIndex, tmp_path: Path) -> None:
    path = tmp_path / "brute.npz"
    index.save(path)
    reloaded = BruteForceIndex.load(path)
    q = unit(np.array([0.2, 0.9, 0.1]))
    assert index.search(q, k=3) == reloaded.search(q, k=3)


def test_neighbor_similarity_is_one_minus_distance() -> None:
    assert Neighbor(id=1, distance=0.25).similarity == pytest.approx(0.75)


def test_intrinsic_dimension_recovers_a_known_manifold() -> None:
    """Points on a uniform 2-d sphere embedded in 64-d must read as ~2, not 64.

    Validating the estimator against data points whose answer you already
    know is what earns you the right to believe it when it says your real
    corpus is near 1-dimensional. A unit-normalized plane collapses to a
    great circle (a 1-d manifold), so this uses a genuinely 2-d construct.
    """
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, np.pi, 4000)
    phi = rng.uniform(0.0, 2.0 * np.pi, 4000)
    sphere = np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    ).astype(np.float32)
    padded = np.zeros((4000, 64), dtype=np.float32)
    padded[:, :3] = sphere
    vectors = unit(padded)
    assert 1.0 < intrinsic_dimension(vectors, sample=800) < 4.0
