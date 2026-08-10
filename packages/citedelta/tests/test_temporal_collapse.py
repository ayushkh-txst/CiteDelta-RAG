"""The collapse measurement, on data where the answer is known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from citedelta.bench.strategies import post_filter_search, required_overfetch
from citedelta.bench.temporal import held_out_split
from citedelta.index.brute import BruteForceIndex
from citedelta.temporal import AdmissibleSet


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


def versioned_corpus(
    n_distinct: int = 60, versions: int = 20, dim: int = 16
) -> tuple[np.ndarray, np.ndarray, set[int]]:
    """Mimic the real corpus: each paragraph exists in many near-identical
    versions, and exactly ONE version of each is in force."""
    rng = np.random.default_rng(0)
    base = unit(rng.standard_normal((n_distinct, dim)).astype(np.float32))
    vectors = unit(
        np.repeat(base, versions, axis=0)
        + 0.001 * rng.standard_normal((n_distinct * versions, dim)).astype(np.float32)
    )
    ids = np.arange(n_distinct * versions, dtype=np.int64)
    in_force = {int(i * versions + versions - 1) for i in range(n_distinct)}  # last version
    return ids, vectors, in_force


def test_post_filter_collapses_on_a_versioned_corpus() -> None:
    """The project's core claim, reproduced on synthetic data.

    Selectivity is 1/20 = 5%. A naive top-10 post-filter should recover almost
    nothing, because the 10 nearest neighbours are 10 versions of the SAME
    paragraph and only one is in force.
    """
    ids, vectors, in_force = versioned_corpus()
    ix = BruteForceIndex()
    ix.build(ids, vectors)
    adm = AdmissibleSet(ids=frozenset(in_force), label="test", corpus_size=len(ids))
    mask = ix.compile_filter(adm.ids)
    assert adm.selectivity == pytest.approx(0.05)

    rng = np.random.default_rng(1)
    queries = unit(rng.standard_normal((40, 16)).astype(np.float32))

    naive = oracle_hits = 0
    for q in queries:
        truth = {n.id for n in ix.search(q, 10, admissible=mask)}
        got = {n.id for n in post_filter_search(ix, q, 10, adm, overfetch=1)}
        naive += len(got & truth)
        oracle_hits += len(truth)

    assert oracle_hits == 400, "the filtered oracle should always fill k"
    assert naive / 400 < 0.35, "post-filter should collapse; it did not"


def test_overfetch_recovers_recall() -> None:
    """Enough overfetch restores recall — the cost, not the correctness, is
    the problem."""
    ids, vectors, in_force = versioned_corpus()
    ix = BruteForceIndex()
    ix.build(ids, vectors)
    adm = AdmissibleSet(ids=frozenset(in_force), label="t", corpus_size=len(ids))
    mask = ix.compile_filter(adm.ids)

    rng = np.random.default_rng(2)
    queries = unit(rng.standard_normal((30, 16)).astype(np.float32))

    scores = []
    for overfetch in (1, 10, 100):
        total = 0
        for q in queries:
            truth = {n.id for n in ix.search(q, 10, admissible=mask)}
            got = {n.id for n in post_filter_search(ix, q, 10, adm, overfetch=overfetch)}
            total += len(got & truth)
        scores.append(total / (len(queries) * 10))

    assert scores == sorted(scores), f"recall not monotonic in overfetch: {scores}"
    assert scores[-1] > 0.95


def test_filtered_oracle_always_fills_k_when_enough_are_admissible() -> None:
    ids, vectors, in_force = versioned_corpus()
    ix = BruteForceIndex()
    ix.build(ids, vectors)
    mask = ix.compile_filter(in_force)
    q = unit(np.random.default_rng(3).standard_normal(16).astype(np.float32))
    assert len(ix.search(q, 10, admissible=mask)) == 10


def test_required_overfetch_is_one_over_selectivity() -> None:
    assert required_overfetch(0.5) == 2
    assert required_overfetch(0.02) == 50
    assert required_overfetch(0.0) == 512  # degenerate, capped
    assert required_overfetch(1.0) == 1


def test_held_out_queries_are_absent_from_the_corpus() -> None:
    rng = np.random.default_rng(4)
    vectors = unit(rng.standard_normal((500, 8)).astype(np.float32))
    ids = np.arange(500, dtype=np.int64)
    corpus_ids, corpus_vectors, queries = held_out_split(ids, vectors, n_queries=50)

    assert len(corpus_ids) == 450
    assert len(queries) == 50
    rows = {v.tobytes() for v in corpus_vectors}
    assert all(q.tobytes() not in rows for q in queries)
