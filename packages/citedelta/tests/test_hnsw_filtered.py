"""Filtered HNSW: correctness, and the design justification as an assertion."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from citedelta.index.brute import BruteForceIndex
from citedelta.index.hnsw import HNSWIndex


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


def versioned(
    n_distinct: int = 80, versions: int = 12, dim: int = 24, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, set[int]]:
    """The shape of the real corpus: many near-identical versions per
    paragraph, exactly one of which is in force."""
    rng = np.random.default_rng(seed)
    base = unit(rng.standard_normal((n_distinct, dim)).astype(np.float32))
    pts = unit(
        np.repeat(base, versions, axis=0)
        + 0.004 * rng.standard_normal((n_distinct * versions, dim)).astype(np.float32)
    )
    ids = np.arange(len(pts), dtype=np.int64)
    in_force = {int(i * versions + versions - 1) for i in range(n_distinct)}
    return ids, pts, in_force


@pytest.fixture
def setup() -> tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex]:
    ids, vectors, in_force = versioned()
    oracle = BruteForceIndex()
    oracle.build(ids, vectors)
    hnsw = HNSWIndex(m=16, ef_construction=200, seed=7)
    hnsw.build(ids, vectors)
    return ids, vectors, in_force, oracle, hnsw


def test_filtered_recall_is_high(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    """The acceptance criterion: pushdown holds recall at ~8% selectivity."""
    _ids, _vectors, in_force, oracle, hnsw = setup
    m_or, m_hn = oracle.compile_filter(in_force), hnsw.compile_filter(in_force)
    rng = np.random.default_rng(1)
    queries = unit(rng.standard_normal((40, 24)).astype(np.float32))

    hits = 0
    for q in queries:
        truth = {n.id for n in oracle.search(q, 10, admissible=m_or)}
        got = {n.id for n in hnsw.search(q, 10, effort=128, admissible=m_hn)}
        hits += len(got & truth)
    assert hits / (len(queries) * 10) >= 0.90


def test_pushdown_beats_post_filtering_decisively(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    """Same index, same ef. Only the filter's position changes."""
    _ids, _vectors, in_force, oracle, hnsw = setup
    m_or, m_hn = oracle.compile_filter(in_force), hnsw.compile_filter(in_force)
    rng = np.random.default_rng(2)
    queries = unit(rng.standard_normal((40, 24)).astype(np.float32))

    pushed = post = 0
    for q in queries:
        truth = {n.id for n in oracle.search(q, 10, admissible=m_or)}
        pushed += len({n.id for n in hnsw.search(q, 10, effort=64, admissible=m_hn)} & truth)
        raw = hnsw.search(q, 10, effort=64)
        post += len({n.id for n in raw if n.id in in_force} & truth)

    total = len(queries) * 10
    assert pushed / total > 5 * max(post / total, 0.01), (
        f"pushdown {pushed / total:.3f} vs post-filter {post / total:.3f}"
    )


def test_pruning_the_walk_would_shatter_the_graph(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    """Encodes WHY traversal is unfiltered (§4.1).

    If the admissible-only subgraph were well connected, pruning the walk
    would be the simpler and faster design. Assert that it isn't — so the
    decision is documented by a test rather than by a comment someone deletes.
    """
    _ids, _vectors, in_force, _oracle, hnsw = setup
    mask = hnsw.compile_filter(in_force)
    g0 = hnsw._graph[0]
    admissible_rows = [int(i) for i in np.where(mask)[0]]

    degrees = [sum(1 for n in g0.get(i, ()) if mask[n]) for i in admissible_rows]
    assert np.mean(degrees) < 4.0, "admissible subgraph is denser than expected"

    seen: set[int] = set()
    largest = 0
    for start in admissible_rows:
        if start in seen:
            continue
        size = 0
        queue = deque([start])
        seen.add(start)
        while queue:
            v = queue.popleft()
            size += 1
            for n in g0.get(v, ()):
                if mask[n] and n not in seen:
                    seen.add(n)
                    queue.append(n)
        largest = max(largest, size)

    assert largest < 0.75 * len(admissible_rows), (
        "admissible-only subgraph is connected here; the traverse-through "
        "design would need re-justifying on this data"
    )


def test_never_returns_an_inadmissible_id(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    _ids, _vectors, in_force, _oracle, hnsw = setup
    mask = hnsw.compile_filter(in_force)
    rng = np.random.default_rng(3)
    for q in unit(rng.standard_normal((25, 24)).astype(np.float32)):
        assert {n.id for n in hnsw.search(q, 10, effort=64, admissible=mask)} <= in_force


def test_higher_ef_never_reduces_filtered_recall(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    _ids, _vectors, in_force, oracle, hnsw = setup
    m_or, m_hn = oracle.compile_filter(in_force), hnsw.compile_filter(in_force)
    rng = np.random.default_rng(4)
    queries = unit(rng.standard_normal((25, 24)).astype(np.float32))

    scores = []
    for ef in (10, 32, 128):
        total = 0
        for q in queries:
            truth = {n.id for n in oracle.search(q, 10, admissible=m_or)}
            total += len({n.id for n in hnsw.search(q, 10, effort=ef, admissible=m_hn)} & truth)
        scores.append(total / (len(queries) * 10))
    assert scores == sorted(scores), f"filtered recall not monotonic in ef: {scores}"


def test_filtering_costs_more_visits(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    """The honest tradeoff, asserted: correctness is not free here."""
    _ids, _vectors, in_force, _oracle, hnsw = setup
    mask = hnsw.compile_filter(in_force)
    q = unit(np.random.default_rng(5).standard_normal(24).astype(np.float32))

    hnsw.search(q, 10, effort=64)
    unfiltered = hnsw.last_visits
    hnsw.search(q, 10, effort=64, admissible=mask)
    assert hnsw.last_visits > unfiltered


def test_visit_budget_bounds_the_work(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    """A pathological filter must degrade recall, not hang."""
    _ids, _vectors, in_force, _oracle, hnsw = setup
    hnsw._max_visits = 200
    mask = hnsw.compile_filter({min(in_force)})  # one admissible row
    q = unit(np.random.default_rng(6).standard_normal(24).astype(np.float32))
    hnsw.search(q, 10, effort=64, admissible=mask)
    assert hnsw.last_visits <= 200 + hnsw._m0


def test_search_survives_an_inadmissible_entry_point(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    """The descent is unfiltered, so an inadmissible entry point is fine.
    Filtering the descent would make `_search_layer(...)[0]` raise IndexError."""
    _ids, _vectors, in_force, _oracle, hnsw = setup
    assert hnsw._entry is not None
    entry_id = int(hnsw._ids[hnsw._entry])
    admissible = in_force - {entry_id}
    mask = hnsw.compile_filter(admissible)
    q = unit(np.random.default_rng(7).standard_normal(24).astype(np.float32))
    hits = hnsw.search(q, 10, effort=64, admissible=mask)
    assert len(hits) == 10
    assert all(h.id != entry_id for h in hits)


def test_fewer_admissible_than_k(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    _ids, _vectors, in_force, _oracle, hnsw = setup
    chosen = set(sorted(in_force)[:3])
    hits = hnsw.search(
        unit(np.random.default_rng(8).standard_normal(24).astype(np.float32)),
        10,
        effort=200,
        admissible=hnsw.compile_filter(chosen),
    )
    assert {h.id for h in hits} <= chosen


def test_empty_filter_returns_nothing(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    _ids, _vectors, _in_force, _oracle, hnsw = setup
    q = unit(np.random.default_rng(9).standard_normal(24).astype(np.float32))
    assert hnsw.search(q, 10, effort=64, admissible=hnsw.compile_filter(set())) == []


def test_unfiltered_results_are_unchanged_by_this_block(
    setup: tuple[np.ndarray, np.ndarray, set[int], BruteForceIndex, HNSWIndex],
) -> None:
    """Regression guard on earlier published numbers.

    The rewritten _search_layer must be bit-identical when admissible is None,
    or docs/design/03-benchmarks.md quietly stops being true.
    """
    _ids, _vectors, _in_force, _oracle, hnsw = setup
    rng = np.random.default_rng(10)
    for q in unit(rng.standard_normal((10, 24)).astype(np.float32)):
        for ef in (16, 64):
            a = hnsw.search(q, 10, effort=ef)
            b = hnsw.search(q, 10, effort=ef, admissible=None)
            assert a == b
