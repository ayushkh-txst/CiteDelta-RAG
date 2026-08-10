"""Invariants EVERY VectorIndex must hold. Add a factory, inherit the suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from citedelta.index.brute import BruteForceIndex
from citedelta.index.hnsw import HNSWIndex
from citedelta.index.ivf import IVFFlatIndex
from citedelta.index.vector import Ids, VectorIndex, Vectors

Factory = Callable[[], VectorIndex]
Corpus = tuple[np.ndarray, np.ndarray]

# `exhaustive` is the effort setting at which the index should be EXACT.
# Approximate indexes are only required to match the oracle when told to try
# as hard as possible; testing them at low effort would be testing the
# approximation, which is the benchmark's job, not conformance's.
INDEXES: list[tuple[str, Factory, int | None]] = [
    ("brute-force", BruteForceIndex, None),
    ("ivf-flat", lambda: IVFFlatIndex(n_lists=8, seed=0), 8),
    # ef high enough to be exhaustive on the 200-vector conformance corpus.
    ("hnsw", lambda: HNSWIndex(m=8, ef_construction=200, seed=1), 200),
]

PARAMS = [pytest.param(f, e, id=name) for name, f, e in INDEXES]

# Indexes with in-index filtering implemented. HNSW joins in Block 4.
FILTER_CAPABLE = [
    pytest.param(BruteForceIndex, None, id="brute-force"),
    pytest.param(lambda: IVFFlatIndex(n_lists=8, seed=0), 8, id="ivf-flat"),
]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


@pytest.fixture
def corpus() -> tuple[Ids, Vectors]:
    rng = np.random.default_rng(7)
    vectors = unit(rng.standard_normal((200, 24)).astype(np.float32))
    vectors[150] = vectors[10]  # a deliberate exact duplicate
    return np.arange(1000, 1200, dtype=np.int64), vectors


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
def test_satisfies_the_protocol(factory: Factory, exhaustive: int | None) -> None:
    assert isinstance(factory(), VectorIndex)


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
def test_reports_its_size_and_dimensions(
    factory: Factory, exhaustive: int | None, corpus: Corpus
) -> None:
    ids, vectors = corpus
    ix = factory()
    ix.build(ids, vectors)
    assert ix.size == 200
    assert ix.dimensions == 24


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
def test_never_returns_an_unknown_id(
    factory: Factory, exhaustive: int | None, corpus: Corpus
) -> None:
    """The failure mode that gap/permutation bugs actually produce, and the
    one that is silent: a plausible ranking of ids that do not exist."""
    ids, vectors = corpus
    ix = factory()
    ix.build(ids, vectors)
    known = set(ids.tolist())
    for q in vectors[:20]:
        assert {h.id for h in ix.search(q, 10, effort=exhaustive)} <= known


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
def test_returns_at_most_k_sorted_ascending(
    factory: Factory, exhaustive: int | None, corpus: Corpus
) -> None:
    ids, vectors = corpus
    ix = factory()
    ix.build(ids, vectors)
    hits = ix.search(vectors[3], 10, effort=exhaustive)
    assert len(hits) == 10
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
def test_k_greater_than_corpus_is_clamped(
    factory: Factory, exhaustive: int | None, corpus: Corpus
) -> None:
    ids, vectors = corpus
    ix = factory()
    ix.build(ids, vectors)
    assert len(ix.search(vectors[0], 500, effort=exhaustive)) == 200


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
def test_empty_index_returns_nothing(factory: Factory, exhaustive: int | None) -> None:
    ix = factory()
    ix.build(np.zeros(0, dtype=np.int64), np.zeros((0, 8), dtype=np.float32))
    assert ix.search(unit(np.ones(8)), 5, effort=exhaustive) == []


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
def test_at_full_effort_matches_the_oracle(
    factory: Factory, exhaustive: int | None, corpus: Corpus
) -> None:
    """Exhaustive settings must reproduce exact search, distance for distance.

    Compared on DISTANCES rather than ids: the corpus contains a duplicate
    vector, so two indexes may legitimately choose different ids at the same
    distance. Asserting on ids would fail for a correct implementation.
    """
    ids, vectors = corpus
    oracle = BruteForceIndex()
    oracle.build(ids, vectors)
    ix = factory()
    ix.build(ids, vectors)

    for q in vectors[:15]:
        mine = [h.distance for h in ix.search(q, 10, effort=exhaustive)]
        truth = [h.distance for h in oracle.search(q, 10)]
        assert mine == pytest.approx(truth, abs=1e-5)


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
def test_save_load_round_trip(
    factory: Factory, exhaustive: int | None, corpus: Corpus, tmp_path: Path
) -> None:
    ids, vectors = corpus
    ix = factory()
    ix.build(ids, vectors)
    path = tmp_path / f"{ix.name}.npz"
    ix.save(path)

    reloaded = type(ix).load(path)
    assert reloaded.size == ix.size
    q = vectors[42]
    assert ix.search(q, 5, effort=exhaustive) == reloaded.search(q, 5, effort=exhaustive)


@pytest.mark.parametrize(("factory", "exhaustive"), PARAMS)
@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    n=st.integers(min_value=1, max_value=120),
    dim=st.integers(min_value=2, max_value=16),
    k=st.integers(min_value=1, max_value=20),
)
def test_ids_are_preserved_across_generated_corpora(
    factory: Factory, exhaustive: int | None, n: int, dim: int, k: int
) -> None:
    """Property: an index may reorder, approximate, or truncate — it may
    never invent."""
    rng = np.random.default_rng(n * 31 + dim)
    vectors = unit(rng.standard_normal((n, dim)).astype(np.float32))
    ids = rng.choice(10_000, n, replace=False).astype(np.int64)

    ix = factory()
    ix.build(ids, vectors)
    hits = ix.search(vectors[0], k, effort=exhaustive)

    assert len(hits) == min(k, n)
    assert {h.id for h in hits} <= set(ids.tolist())
    assert len({h.id for h in hits}) == len(hits)  # no duplicates returned


@pytest.mark.parametrize(("factory", "exhaustive"), FILTER_CAPABLE)
def test_filtered_search_matches_the_filtered_oracle(
    factory: Factory, exhaustive: int | None, corpus: Corpus
) -> None:
    ids, vectors = corpus
    admissible = set(ids[::4].tolist())

    oracle = BruteForceIndex()
    oracle.build(ids, vectors)
    ix = factory()
    ix.build(ids, vectors)

    for q in vectors[:10]:
        mine = [
            h.distance
            for h in ix.search(q, 5, effort=exhaustive, admissible=ix.compile_filter(admissible))
        ]
        truth = [
            h.distance for h in oracle.search(q, 5, admissible=oracle.compile_filter(admissible))
        ]
        assert mine == pytest.approx(truth, abs=1e-5)


@pytest.mark.parametrize(("factory", "exhaustive"), FILTER_CAPABLE)
def test_filtered_search_never_returns_an_inadmissible_id(
    factory: Factory, exhaustive: int | None, corpus: Corpus
) -> None:
    ids, vectors = corpus
    admissible = set(ids[::3].tolist())
    ix = factory()
    ix.build(ids, vectors)
    mask = ix.compile_filter(admissible)
    for q in vectors[:10]:
        assert {h.id for h in ix.search(q, 5, effort=exhaustive, admissible=mask)} <= admissible
