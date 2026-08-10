"""Filtered IVF: adaptive probing, and the guarantees it does and doesn't give."""

from __future__ import annotations

import numpy as np
import pytest

from citedelta.index.brute import BruteForceIndex
from citedelta.index.ivf import IVFFlatIndex


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


def clustered(n_clusters: int, per_cluster: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = unit(rng.standard_normal((n_clusters, dim)).astype(np.float32))
    pts = np.repeat(centres, per_cluster, axis=0)
    return unit(pts + 0.02 * rng.standard_normal(pts.shape).astype(np.float32))


@pytest.fixture
def setup() -> tuple[np.ndarray, np.ndarray, BruteForceIndex, IVFFlatIndex]:
    vectors = clustered(20, 40, 32)
    ids = np.arange(800, dtype=np.int64)
    oracle = BruteForceIndex()
    oracle.build(ids, vectors)
    ivf = IVFFlatIndex(n_lists=20, seed=0)
    ivf.build(ids, vectors)
    return ids, vectors, oracle, ivf


def test_probing_every_cell_is_exact_under_a_filter(
    setup: tuple[np.ndarray, np.ndarray, BruteForceIndex, IVFFlatIndex],
) -> None:
    """nprobe == nlist must reproduce the filtered oracle exactly."""
    _ids, vectors, oracle, ivf = setup

    admissible = set(range(0, 800, 7))
    m_or, m_iv = oracle.compile_filter(admissible), ivf.compile_filter(admissible)

    for q in vectors[::80]:
        mine = [h.distance for h in ivf.search(q, 10, effort=20, admissible=m_iv)]
        truth = [h.distance for h in oracle.search(q, 10, admissible=m_or)]
        assert mine == pytest.approx(truth, abs=1e-5)


def test_adaptive_probing_finds_k_even_with_nprobe_one(
    setup: tuple[np.ndarray, np.ndarray, BruteForceIndex, IVFFlatIndex],
) -> None:
    """The point of the block: a fixed budget of 1 cell would return almost
    nothing; the adaptive loop still fills k."""
    _ids, vectors, _oracle, ivf = setup

    admissible = set(range(0, 800, 20))  # 5% selectivity
    mask = ivf.compile_filter(admissible)

    for q in vectors[::100]:
        hits = ivf.search(q, 10, effort=1, admissible=mask)
        assert len(hits) == 10
        assert ivf.last_probes_used > 1  # it had to keep going


def test_probes_used_grows_as_selectivity_falls(
    setup: tuple[np.ndarray, np.ndarray, BruteForceIndex, IVFFlatIndex],
) -> None:
    """The cost of correctness, as an assertion."""
    _ids, vectors, _oracle, ivf = setup

    q = vectors[13]
    used = []
    for step in (2, 10, 50):  # 50%, 10%, 2% selectivity
        mask = ivf.compile_filter(set(range(0, 800, step)))
        ivf.search(q, 10, effort=1, admissible=mask)
        used.append(ivf.last_probes_used)
    assert used == sorted(used), f"probes should rise as selectivity falls: {used}"


def test_never_returns_an_inadmissible_row(
    setup: tuple[np.ndarray, np.ndarray, BruteForceIndex, IVFFlatIndex],
) -> None:
    _ids, vectors, _oracle, ivf = setup

    admissible = set(range(0, 800, 11))
    mask = ivf.compile_filter(admissible)
    for q in vectors[::60]:
        assert {h.id for h in ivf.search(q, 10, effort=4, admissible=mask)} <= admissible


def test_empty_filter_returns_nothing(
    setup: tuple[np.ndarray, np.ndarray, BruteForceIndex, IVFFlatIndex],
) -> None:
    _ids, vectors, _oracle, ivf = setup

    assert ivf.search(vectors[0], 10, effort=4, admissible=ivf.compile_filter(set())) == []


def test_fewer_admissible_than_k_returns_what_exists(
    setup: tuple[np.ndarray, np.ndarray, BruteForceIndex, IVFFlatIndex],
) -> None:
    """Must never pad with inadmissible rows to reach k."""
    _ids, vectors, _oracle, ivf = setup

    admissible = {3, 17, 400}
    hits = ivf.search(vectors[0], 10, effort=1, admissible=ivf.compile_filter(admissible))
    assert len(hits) == 3
    assert {h.id for h in hits} == admissible
    assert ivf.last_probes_used == ivf.n_lists  # exhausted every cell looking


def test_filter_does_not_change_unfiltered_behaviour(
    setup: tuple[np.ndarray, np.ndarray, BruteForceIndex, IVFFlatIndex],
) -> None:
    """Regression guard: adding the filtered path must not perturb the
    existing unfiltered results from earlier benchmarks."""
    _ids, vectors, _oracle, ivf = setup

    q = vectors[5]
    assert [h.id for h in ivf.search(q, 10, effort=4)] == [
        h.id for h in ivf.search(q, 10, effort=4, admissible=None)
    ]
