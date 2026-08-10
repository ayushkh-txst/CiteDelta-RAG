"""The index, verified against an implementation too simple to be wrong."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from citedelta.index.lexical import K1, B, LexicalIndex, build_index
from citedelta.index.tokenize import tokenize
from citedelta.index.varint import read_varint, write_varint

# --------------------------------------------------------------- varint


@given(values=st.lists(st.integers(min_value=0, max_value=2**63 - 1), max_size=200))
def test_varint_round_trips(values: list[int]) -> None:
    buf = bytearray()
    for v in values:
        write_varint(v, buf)
    pos = 0
    out = []
    for _ in values:
        v, pos = read_varint(buf, pos)
        out.append(v)
    assert out == values
    assert pos == len(buf)  # nothing left over, nothing over-read


@given(value=st.integers(min_value=0, max_value=127))
def test_small_values_cost_one_byte(value: int) -> None:
    buf = bytearray()
    write_varint(value, buf)
    assert len(buf) == 1


def test_negative_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsigned"):
        write_varint(-1, bytearray())


# ------------------------------------------------------------- tokenizer


def test_legal_identifiers_survive() -> None:
    toks = tokenize("See 8 CFR 214.2(f)(10)(ii)(C) and Form I-20 for F-1 status.")
    assert "214.2" in toks
    assert "i-20" in toks
    assert "f-1" in toks


def test_normalization_unifies_variants() -> None:
    assert tokenize("Ｆ-1 STATUS") == tokenize("f-1 status")  # noqa: RUF001 - the fullwidth F is the point


# ------------------------------------------------------------ the oracle


def brute_force_search(docs: list[tuple[int, str]], query: str, k: int) -> list[tuple[int, float]]:
    """BM25 with no index at all. Slow, obvious, and the source of truth."""
    tokenized = [(cid, tokenize(text)) for cid, text in docs]
    n = len(tokenized)
    avgdl = sum(len(t) for _, t in tokenized) / n

    df: Counter[str] = Counter()
    for _, terms in tokenized:
        df.update(set(terms))

    scored: list[tuple[int, float, int]] = []
    for i, (cid, terms) in enumerate(tokenized):
        counts = Counter(terms)
        total = 0.0
        for term in tokenize(query):
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * len(terms) / avgdl))
        if total > 0:
            scored.append((cid, total, i))

    scored.sort(key=lambda t: (-t[1], t[2]))
    return [(cid, score) for cid, score, _ in scored[:k]]


CORPUS = [
    (101, "Optional practical training may be authorized for an F-1 student."),
    (102, "Curricular practical training is an integral part of the curriculum."),
    (103, "A STEM extension of post-completion optional practical training."),
    (104, "The student must not exceed 90 days of unemployment."),
    (105, "See 8 CFR 214.2(f)(10)(ii)(C) for the 24-month extension."),
    (106, "Nonimmigrant classes include foreign government officials."),
    (
        107,
        "It is the position of the administrative office that the term practical "
        "training may not be construed as employment in general or as an other "
        "form without the approval of the responsible official and the consent "
        "of the beneficiary in accordance with the procedures established for "
        "adjudication on a case by case basis.",
    ),
    (108, "An aggregate of 150 days of unemployment during the extension."),
]


@pytest.fixture
def index(tmp_path: Path) -> Iterator[LexicalIndex]:
    build_index(CORPUS, tmp_path / "t.idx")
    with LexicalIndex(tmp_path / "t.idx") as ix:
        yield ix


@pytest.mark.parametrize(
    "query",
    [
        "optional practical training",
        "unemployment",
        "214.2",
        "stem extension",
        "practical training extension unemployment",
        "f-1 student",
    ],
)
def test_index_matches_the_oracle(index: LexicalIndex, query: str) -> None:
    mine = [(h.chunk_id, h.score) for h in index.search(query, k=5)]
    theirs = brute_force_search(CORPUS, query, k=5)

    assert [c for c, _ in mine] == [c for c, _ in theirs], query
    for (_, a), (_, b) in zip(mine, theirs, strict=True):
        assert a == pytest.approx(b, rel=1e-9)


def test_length_normalization_penalizes_padding(index: LexicalIndex) -> None:
    """A short focused chunk beats a padding chunk that repeats the terms."""
    ranked = [h.chunk_id for h in index.search("practical training", k=8)]
    assert ranked[0] in (101, 102)


def test_unknown_terms_and_empty_queries_are_safe(index: LexicalIndex) -> None:
    assert index.search("zzzzz") == []
    assert index.search("") == []
    assert index.search("the and of") == []  # all stopwords


def test_postings_are_sorted_and_complete(index: LexicalIndex) -> None:
    docs = [d for d, _ in index.postings("practical")]
    assert docs == sorted(docs)
    assert len(docs) == len(set(docs))


def test_allowed_filter_restricts_results(index: LexicalIndex) -> None:
    hits = index.search("practical training", k=8, admissible=index.compile_filter({102, 103}))
    assert {h.chunk_id for h in hits} <= {102, 103}


def test_filtered_search_is_exact(tmp_path: Path) -> None:
    """Filtering before top-k gives the k best ADMISSIBLE docs — verified
    against an index built only from the admissible documents."""
    build_index(CORPUS, tmp_path / "all.idx")
    admissible = {102, 103, 108}
    build_index([d for d in CORPUS if d[0] in admissible], tmp_path / "sub.idx")

    with LexicalIndex(tmp_path / "all.idx") as full, LexicalIndex(tmp_path / "sub.idx") as sub:
        mask = full.compile_filter(admissible)
        for query in ("practical training", "unemployment", "extension"):
            mine = [h.chunk_id for h in full.search(query, 5, admissible=mask)]
            # Same documents, same order. (Scores differ: IDF is computed over
            # the full corpus in one and the subset in the other — the
            # deliberate approximation documented in the search docstring.)
            assert mine == [h.chunk_id for h in sub.search(query, 5)]


def test_post_filtering_after_topk_loses_documents(tmp_path: Path) -> None:
    """The core claim, on the lexical side. Same ranker, filter moved."""
    build_index(CORPUS, tmp_path / "t.idx")
    admissible = {108}  # only the least-similar match survives
    with LexicalIndex(tmp_path / "t.idx") as ix:
        mask = ix.compile_filter(admissible)
        exact = ix.search("practical training unemployment", 3, admissible=mask)
        naive = [
            h for h in ix.search("practical training unemployment", 3) if h.chunk_id in admissible
        ]

        assert [h.chunk_id for h in exact] == [108]
        assert naive == []  # post-filter found nothing


def test_filter_never_returns_an_inadmissible_document(tmp_path: Path) -> None:
    build_index(CORPUS, tmp_path / "t.idx")
    admissible = {101, 104}
    with LexicalIndex(tmp_path / "t.idx") as ix:
        mask = ix.compile_filter(admissible)
        for query in ("practical", "unemployment", "student", "extension"):
            assert {h.chunk_id for h in ix.search(query, 10, admissible=mask)} <= admissible


def test_empty_filter_returns_nothing(tmp_path: Path) -> None:
    build_index(CORPUS, tmp_path / "t.idx")
    with LexicalIndex(tmp_path / "t.idx") as ix:
        assert ix.search("practical training", 5, admissible=ix.compile_filter(set())) == []


def test_filter_mask_aligns_with_internal_positions(tmp_path: Path) -> None:
    """The mask indexes internal positions, but callers speak in chunk ids.
    A misalignment here silently returns the wrong documents."""
    build_index(CORPUS, tmp_path / "t.idx")
    with LexicalIndex(tmp_path / "t.idx") as ix:
        mask = ix.compile_filter({105})
        assert int(mask.sum()) == 1
        assert int(ix._doc_ids[int(np.argmax(mask))]) == 105


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "x.idx"
    build_index(CORPUS, path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_rebuild_replaces_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "x.idx"
    build_index(CORPUS, path)
    build_index(CORPUS[:3], path)
    with LexicalIndex(path) as ix:
        assert ix.n_docs == 3


def test_rejects_a_foreign_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.idx"
    bogus.write_bytes(b"NOTANIDX" + b"\x00" * 200)
    with pytest.raises(ValueError, match="not a CiteDelta index"):
        LexicalIndex(bogus)


@given(
    docs=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=10_000),
            st.text(alphabet="abcdef ", min_size=1, max_size=60),
        ),
        min_size=1,
        max_size=40,
        unique_by=lambda t: t[0],
    )
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_never_returns_a_document_it_was_not_given(
    docs: list[tuple[int, str]], tmp_path: Path
) -> None:
    """The index conformance invariant held across generated corpora.

    Gap decoding that goes wrong silently invents document ids; an index that
    does so must fail here.
    """
    path = tmp_path / f"h{len(docs)}.idx"
    build_index(docs, path)
    given_ids = {cid for cid, _ in docs}
    with LexicalIndex(path) as ix:
        for hit in ix.search("abc def", k=10):
            assert hit.chunk_id in given_ids
