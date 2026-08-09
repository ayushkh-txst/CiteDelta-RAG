"""The harness measures what it claims to."""

from __future__ import annotations

import numpy as np
import pytest

from citedelta.bench.datasets import Dataset, cfr_dedup, cfr_full, random_hard
from citedelta.bench.metrics import compute_ground_truth, recall_by_id, recall_with_ties
from citedelta.bench.runner import measure
from citedelta.index.brute import BruteForceIndex
from citedelta.index.vector import Neighbor


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.linalg.norm(v, axis=-1, keepdims=True))


def test_ground_truth_matches_the_oracle() -> None:
    """Two independent paths to the same answer must agree."""
    rng = np.random.default_rng(0)
    vectors = unit(rng.standard_normal((300, 16)).astype(np.float32))
    ids = np.arange(300, dtype=np.int64) * 7
    queries = unit(rng.standard_normal((20, 16)).astype(np.float32))

    truth = compute_ground_truth(vectors, ids, queries, k=5)

    index = BruteForceIndex()
    index.build(ids, vectors)
    for i, q in enumerate(queries):
        assert [h.id for h in index.search(q, 5)] == truth.ids[i].tolist()


def test_exact_search_scores_perfect_recall() -> None:
    """The harness's self-test. If this is not 1.0, the harness is broken."""
    rng = np.random.default_rng(1)
    vectors = unit(rng.standard_normal((500, 32)).astype(np.float32))
    ids = np.arange(500, dtype=np.int64)
    ds = Dataset(
        name="t",
        ids=ids,
        vectors=vectors,
        queries=unit(rng.standard_normal((25, 32)).astype(np.float32)),
    )
    truth = compute_ground_truth(ds.vectors, ds.ids, ds.queries, k=10)

    index = BruteForceIndex()
    index.build(ds.ids, ds.vectors)
    assert measure(index, ds, truth, k=10).recall == pytest.approx(1.0)


def test_tie_aware_recall_forgives_a_different_but_equal_choice() -> None:
    """The core metric decision, as an assertion."""
    truth_ids = np.array([1, 2, 3], dtype=np.int64)
    truth_dist = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    # Ids 7,8,9 are different rows at IDENTICAL distance — equally correct.
    retrieved = [Neighbor(7, 0.1), Neighbor(8, 0.1), Neighbor(9, 0.1)]

    assert recall_with_ties(retrieved, truth_dist, 3) == 1.0
    assert recall_by_id(retrieved, truth_ids, 3) == 0.0  # naive metric: total miss


def test_recall_still_punishes_genuinely_worse_results() -> None:
    """Tie tolerance must not become 'everything passes'."""
    truth_dist = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    retrieved = [Neighbor(7, 0.1), Neighbor(8, 0.9), Neighbor(9, 0.9)]
    assert recall_with_ties(retrieved, truth_dist, 3) == pytest.approx(1 / 3)


def test_queries_are_held_out_of_the_index() -> None:
    """No query may appear in the corpus it searches, or self-match skews all."""
    rng = np.random.default_rng(2)
    vectors = unit(rng.standard_normal((1000, 8)).astype(np.float32))
    ds = cfr_full(np.arange(1000, dtype=np.int64), vectors)

    assert ds.size == 1000 - len(ds.queries)
    corpus_rows = {v.tobytes() for v in ds.vectors}
    assert all(q.tobytes() not in corpus_rows for q in ds.queries)


def test_dedup_dataset_removes_duplicate_vectors() -> None:
    base = unit(np.random.default_rng(3).standard_normal((50, 8)).astype(np.float32))
    vectors = np.vstack([base, base, base])  # every vector three times
    ids = np.arange(150, dtype=np.int64)

    assert cfr_full(ids, vectors).size > cfr_dedup(ids, vectors).size
    deduped = cfr_dedup(ids, vectors)
    assert len(np.unique(deduped.vectors, axis=0)) == len(deduped.vectors)


def test_random_hard_is_actually_hard() -> None:
    """Sanity-check the premise: random high-dim points are near-equidistant."""
    ds = random_hard(n=2000, dim=128)
    sims = ds.vectors[:200] @ ds.vectors.T
    spread = float(np.std(sims))
    assert spread < 0.15, "random vectors should concentrate; they are not concentrating"
