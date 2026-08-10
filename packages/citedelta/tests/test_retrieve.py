"""The hybrid path enforces the temporal predicate on BOTH retrievers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from citedelta.index.brute import BruteForceIndex
from citedelta.index.lexical import LexicalIndex, build_index
from citedelta.index.vector import Vectors
from citedelta.retrieve import hybrid_search
from citedelta.temporal import AdmissibleSet

DOCS = [
    (1, "Optional practical training may be authorized for an F-1 student."),
    (2, "A STEM extension of post-completion optional practical training."),
    (3, "The student must not exceed 90 days of unemployment."),
    (4, "An aggregate of 150 days of unemployment during the extension."),
    (5, "Foreign government officials are admitted under section 214.2(a)."),
]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


@pytest.fixture
def indexes(
    tmp_path: Path,
) -> Iterator[tuple[LexicalIndex, BruteForceIndex, np.ndarray]]:
    build_index(DOCS, tmp_path / "lex.idx")
    rng = np.random.default_rng(0)
    vectors: Vectors = unit(rng.standard_normal((len(DOCS), 16)).astype(np.float32))
    vector = BruteForceIndex()
    vector.build(np.array([d[0] for d in DOCS], dtype=np.int64), vectors)
    with LexicalIndex(tmp_path / "lex.idx") as lexical:
        yield lexical, vector, vectors


def test_no_inadmissible_chunk_can_appear(
    indexes: tuple[LexicalIndex, BruteForceIndex, np.ndarray],
) -> None:
    lexical, vector, vectors = indexes
    admissible = AdmissibleSet(ids=frozenset({3, 5}), label="t", corpus_size=len(DOCS))
    trace = hybrid_search(
        "unemployment practical training",
        vectors[0],
        lexical=lexical,
        vector=vector,
        admissible=admissible,
        k=5,
    )
    assert {h.chunk_id for h in trace.hits} <= {3, 5}


def test_trace_records_both_retrievers(
    indexes: tuple[LexicalIndex, BruteForceIndex, np.ndarray],
) -> None:
    lexical, vector, vectors = indexes
    admissible = AdmissibleSet(
        ids=frozenset(d[0] for d in DOCS), label="all", corpus_size=len(DOCS)
    )
    trace = hybrid_search(
        "unemployment",
        vectors[2],
        lexical=lexical,
        vector=vector,
        admissible=admissible,
        k=3,
    )
    assert trace.candidates_lexical > 0
    assert trace.candidates_vector > 0
    assert any("lexical" in h.ranks or "vector" in h.ranks for h in trace.hits)


def test_empty_admissible_set_yields_nothing(
    indexes: tuple[LexicalIndex, BruteForceIndex, np.ndarray],
) -> None:
    lexical, vector, vectors = indexes
    trace = hybrid_search(
        "unemployment",
        vectors[0],
        lexical=lexical,
        vector=vector,
        admissible=AdmissibleSet(frozenset(), "none", len(DOCS)),
        k=5,
    )
    assert trace.hits == []


def test_a_chunk_found_by_only_one_retriever_still_surfaces(
    indexes: tuple[LexicalIndex, BruteForceIndex, np.ndarray],
) -> None:
    """Fusion must not require agreement — a lexical-only exact match is
    exactly the case hybrid search exists to keep."""
    lexical, vector, vectors = indexes
    admissible = AdmissibleSet(
        ids=frozenset(d[0] for d in DOCS), label="all", corpus_size=len(DOCS)
    )
    trace = hybrid_search(
        "214.2(a)",
        vectors[0],
        lexical=lexical,
        vector=vector,
        admissible=admissible,
        k=5,
    )
    assert 5 in {h.chunk_id for h in trace.hits}
