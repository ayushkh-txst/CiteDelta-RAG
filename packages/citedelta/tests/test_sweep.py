"""The sweep's own invariants."""

from __future__ import annotations

import numpy as np
import pytest

from citedelta.bench.temporal import synthetic_admissible


def test_synthetic_admissible_hits_the_requested_selectivity() -> None:
    ids = np.arange(10_000, dtype=np.int64)
    for target in (0.5, 0.05, 0.005):
        adm = synthetic_admissible(ids, target)
        assert adm.selectivity == pytest.approx(target, rel=0.02)


def test_synthetic_admissible_is_reproducible() -> None:
    ids = np.arange(1_000, dtype=np.int64)
    assert synthetic_admissible(ids, 0.1, seed=3).ids == synthetic_admissible(ids, 0.1, seed=3).ids


def test_synthetic_admissible_never_empty() -> None:
    """Degenerate selectivity must still yield a usable filter."""
    ids = np.arange(100, dtype=np.int64)
    assert synthetic_admissible(ids, 0.0001).size >= 1


def test_synthetic_ids_are_real_corpus_ids() -> None:
    ids = np.array([7, 19, 23, 88], dtype=np.int64)
    assert synthetic_admissible(ids, 0.5).ids <= set(ids.tolist())
