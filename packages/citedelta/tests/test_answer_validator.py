from __future__ import annotations

from citedelta.answer.validator import validate_citations
from citedelta.temporal import AdmissibleSet


def _admissible(ids: set[int]) -> AdmissibleSet:
    return AdmissibleSet(ids=frozenset(ids), label="2026-08-11", corpus_size=100)


def test_clean_citations_pass() -> None:
    r = validate_citations([1, 2], retrieved_ids={1, 2, 3}, admissible=_admissible({1, 2, 3}))
    assert r.ok
    assert r.cited == (1, 2)


def test_invented_id_fails_the_retrieved_check() -> None:
    r = validate_citations([1, 999], retrieved_ids={1, 2}, admissible=_admissible({1, 2}))
    assert not r.ok
    assert r.failures[0].chunk_id == 999
    assert r.failures[0].check == "retrieved"


def test_id_outside_the_corpus_fails_the_exists_check_first() -> None:
    r = validate_citations(
        [42],
        retrieved_ids={1},
        admissible=_admissible({1}),
        corpus_ids={1, 2, 3},
    )
    assert r.failures[0].check == "exists"


def test_retrieved_but_inadmissible_is_caught() -> None:
    """This can only happen if the temporal filter itself is broken. The
    check exists precisely so that bug surfaces as a refusal rather than as
    a confident answer about what the law said."""
    r = validate_citations([7], retrieved_ids={7}, admissible=_admissible({1, 2}))
    assert not r.ok
    assert r.failures[0].check == "admissible"


def test_duplicate_citations_are_deduped_not_double_reported() -> None:
    r = validate_citations([3, 3, 3], retrieved_ids={3}, admissible=_admissible({3}))
    assert r.ok
    assert r.cited == (3,)
