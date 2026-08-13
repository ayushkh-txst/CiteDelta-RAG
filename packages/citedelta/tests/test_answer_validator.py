from __future__ import annotations

from citedelta.answer.models import Citation
from citedelta.answer.validator import CitedRef, validate_citations
from citedelta.temporal import AdmissibleSet


def _admissible(ids: set[int]) -> AdmissibleSet:
    return AdmissibleSet(ids=frozenset(ids), label="2026-08-11", corpus_size=100)


def _src(chunk_id: int, text: str) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        citation_path=f"8 CFR 214.2(f)({chunk_id})",
        effective_from="2016-01-01",
        effective_to=None,
        text=text,
        rrf_score=0.03,
    )


def _refs(ids: list[int]) -> list[CitedRef]:
    return [CitedRef(chunk_id=i, quote="the words are here") for i in ids]


def test_clean_citations_pass() -> None:
    r = validate_citations(
        _refs([1, 2]),
        retrieved={1: _src(1, "the words are here"), 2: _src(2, "the words are here")},
        admissible=_admissible({1, 2, 3}),
    )
    assert r.ok
    assert r.cited == (1, 2)


def test_invented_id_fails_the_retrieved_check() -> None:
    r = validate_citations(
        _refs([1, 999]),
        retrieved={1: _src(1, "the words are here")},
        admissible=_admissible({1, 2}),
    )
    assert not r.ok
    assert r.failures[0].chunk_id == 999
    assert r.failures[0].check == "retrieved"


def test_id_outside_the_corpus_fails_the_exists_check_first() -> None:
    r = validate_citations(
        _refs([42]),
        retrieved={1: _src(1, "the words are here")},
        admissible=_admissible({1}),
        corpus_ids={1, 2, 3},
    )
    assert r.failures[0].check == "exists"


def test_retrieved_but_inadmissible_is_caught() -> None:
    """This can only happen if the temporal filter itself is broken. The
    check exists precisely so that bug surfaces as a refusal rather than as
    a confident answer about what the law said."""
    r = validate_citations(
        _refs([7]),
        retrieved={7: _src(7, "the words are here")},
        admissible=_admissible({1, 2}),
    )
    assert not r.ok
    assert r.failures[0].check == "admissible"


def test_duplicate_citations_are_deduped_not_double_reported() -> None:
    r = validate_citations(
        _refs([3, 3, 3]),
        retrieved={3: _src(3, "the words are here")},
        admissible=_admissible({3}),
    )
    assert r.ok
    assert r.cited == (3,)


CHUNK = (
    "(iv) Preparation for departure. An F-1 student who has completed a\n"
    "course of study will be allowed an additional 60-day period to prepare\n"
    "for departure from the United States."
)


def test_verbatim_quote_passes() -> None:
    r = validate_citations(
        [CitedRef(1, "will be allowed an additional 60-day period")],
        retrieved={1: _src(1, CHUNK)},
        admissible=_admissible({1}),
    )
    assert r.ok


def test_reflowed_whitespace_still_passes() -> None:
    """The source has XML line breaks; a faithful copy reflows them. This is
    the one difference that is never semantic."""
    r = validate_citations(
        [CitedRef(1, "completed a course of study")],  # spans a newline in CHUNK
        retrieved={1: _src(1, CHUNK)},
        admissible=_admissible({1}),
    )
    assert r.ok


def test_one_altered_word_fails() -> None:
    r = validate_citations(
        [CitedRef(1, "will be allowed an additional 90-day period")],
        retrieved={1: _src(1, CHUNK)},
        admissible=_admissible({1}),
    )
    assert not r.ok
    assert r.failures[0].check == "quote"


def test_paraphrase_fails() -> None:
    """The failure mode the check exists for: fluent, accurate in substance,
    and not what the source says."""
    r = validate_citations(
        [CitedRef(1, "students get an extra 60 days to leave")],
        retrieved={1: _src(1, CHUNK)},
        admissible=_admissible({1}),
    )
    assert not r.ok
    assert r.failures[0].check == "quote"


def test_missing_quote_fails() -> None:
    r = validate_citations(
        [CitedRef(1, "  ")],
        retrieved={1: _src(1, CHUNK)},
        admissible=_admissible({1}),
    )
    assert r.failures[0].check == "quote"


def test_quote_from_the_wrong_chunk_fails() -> None:
    """Real text, real id, wrong pairing — caught because the check is per
    citation, not against the union of everything retrieved."""
    r = validate_citations(
        [CitedRef(2, "will be allowed an additional 60-day period")],
        retrieved={1: _src(1, CHUNK), 2: _src(2, "(i) General. Unrelated text.")},
        admissible=_admissible({1, 2}),
    )
    assert not r.ok
    assert r.failures[0].check == "quote"


def test_case_change_fails() -> None:
    r = validate_citations(
        [CitedRef(1, "Will Be Allowed An Additional 60-Day Period")],
        retrieved={1: _src(1, CHUNK)},
        admissible=_admissible({1}),
    )
    assert not r.ok
