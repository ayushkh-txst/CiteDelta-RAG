"""Structural invariants of the graph itself."""

from __future__ import annotations

import math

import numpy as np
import pytest

from citedelta.index.brute import BruteForceIndex
from citedelta.index.hnsw import HNSWIndex


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


@pytest.fixture
def built() -> HNSWIndex:
    rng = np.random.default_rng(3)
    centres = unit(rng.standard_normal((12, 32)).astype(np.float32))
    pts = unit(
        np.repeat(centres, 50, axis=0) + 0.05 * rng.standard_normal((600, 32)).astype(np.float32)
    )
    ix = HNSWIndex(m=8, ef_construction=100, seed=1)
    ix.build(np.arange(600, dtype=np.int64), pts)
    return ix


def test_edges_are_bidirectional(built: HNSWIndex) -> None:
    """If a->b exists then b->a must. Otherwise nodes become unreachable."""
    for layer, adjacency in enumerate(built._graph):
        for node, neighbours in adjacency.items():
            for neighbour in neighbours:
                assert node in built._graph[layer].get(neighbour, []), (
                    f"layer {layer}: {node}->{neighbour} has no reverse edge"
                )


def test_no_self_loops(built: HNSWIndex) -> None:
    for adjacency in built._graph:
        for node, neighbours in adjacency.items():
            assert node not in neighbours


def test_degree_stays_within_budget(built: HNSWIndex) -> None:
    """Pruning must actually fire, or memory grows without bound."""
    for layer, adjacency in enumerate(built._graph):
        budget = built._m0 if layer == 0 else built._m
        for neighbours in adjacency.values():
            assert len(neighbours) <= budget


def test_every_node_is_present_on_layer_zero(built: HNSWIndex) -> None:
    assert set(built._graph[0]) == set(range(built.size))


def test_layer_populations_decay_geometrically(built: HNSWIndex) -> None:
    """Each layer should hold roughly 1/M of the one below (§5.2)."""
    sizes = [len(layer) for layer in built._graph]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == built.size


def test_max_level_is_logarithmic(built: HNSWIndex) -> None:
    expected = math.log(built.size) / math.log(built._m)
    assert built.max_level <= expected + 3


def test_ef_search_is_clamped_to_k(built: HNSWIndex) -> None:
    """Asking for 20 with ef=1 must still return 20, not 1."""
    q = unit(np.random.default_rng(9).standard_normal(32).astype(np.float32))
    assert len(built.search(q, 20, effort=1)) == 20


def test_higher_ef_never_reduces_recall(built: HNSWIndex) -> None:
    rng = np.random.default_rng(11)
    queries = unit(rng.standard_normal((30, 32)).astype(np.float32))
    oracle = BruteForceIndex()
    oracle.build(np.arange(built.size, dtype=np.int64), built._vectors)

    scores = []
    for ef in (10, 25, 100):
        total = 0.0
        for q in queries:
            cutoff = oracle.search(q, 10)[-1].distance + 1e-5
            total += sum(1 for h in built.search(q, 10, effort=ef) if h.distance <= cutoff) / 10
        scores.append(total / len(queries))

    assert scores == sorted(scores), f"recall not monotonic in ef: {scores}"
    assert scores[-1] > 0.9


def test_build_is_reproducible() -> None:
    """Same seed, same graph — or the benchmark isn't reproducible either."""
    rng = np.random.default_rng(5)
    vectors = unit(rng.standard_normal((300, 16)).astype(np.float32))
    ids = np.arange(300, dtype=np.int64)

    a = HNSWIndex(m=8, seed=99)
    a.build(ids, vectors)
    b = HNSWIndex(m=8, seed=99)
    b.build(ids, vectors)

    assert a.edge_count() == b.edge_count()
    q = vectors[7]
    assert a.search(q, 10, effort=50) == b.search(q, 10, effort=50)


def test_single_vector_index() -> None:
    ix = HNSWIndex(m=8)
    ix.build(np.array([42], dtype=np.int64), unit(np.ones((1, 8), dtype=np.float32)))
    hits = ix.search(unit(np.ones(8, dtype=np.float32)), k=5)
    assert len(hits) == 1
    assert hits[0].id == 42
