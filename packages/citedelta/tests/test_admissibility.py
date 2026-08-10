"""Admissible sets, filter compilation, and the filtered oracle."""

from __future__ import annotations

from datetime import date
from hashlib import sha256

import numpy as np
import pytest

from citedelta.index.brute import BruteForceIndex
from citedelta.index.vector import compile_mask
from citedelta.store.corpus import CorpusStore
from citedelta.temporal import AdmissibleSet, AsOf
from substrate.db import Database


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


# ------------------------------------------------------------- mask compilation


def test_compile_mask_selects_exactly_the_admissible_rows() -> None:
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    mask = compile_mask(ids, {20, 40})
    assert mask.tolist() == [False, True, False, True, False]


def test_compile_mask_ignores_ids_not_in_the_index() -> None:
    ids = np.array([10, 20, 30], dtype=np.int64)
    assert compile_mask(ids, {20, 999}).tolist() == [False, True, False]


def test_compile_mask_handles_an_empty_admissible_set() -> None:
    ids = np.array([10, 20, 30], dtype=np.int64)
    assert not compile_mask(ids, set()).any()


def test_compile_mask_respects_permuted_id_order() -> None:
    """IVF permutes its rows. The mask must follow the index's order, not the
    caller's — this is the invariant that makes filtering index-agnostic."""
    ids = np.array([50, 10, 40, 20, 30], dtype=np.int64)  # as if cluster-ordered
    assert compile_mask(ids, {10, 30}).tolist() == [False, True, False, False, True]


# ---------------------------------------------------------- the filtered oracle


@pytest.fixture
def index() -> BruteForceIndex:
    vectors = unit(
        np.array(
            [[1, 0, 0], [0.95, 0.05, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        )
    )
    ix = BruteForceIndex()
    ix.build(np.array([1, 2, 3, 4, 5], dtype=np.int64), vectors)
    return ix


def test_filter_excludes_the_nearest_when_it_is_inadmissible(index: BruteForceIndex) -> None:
    """The whole point: the true nearest neighbour can be out of force."""
    q = unit(np.array([1.0, 0.0, 0.0]))
    assert index.search(q, 1)[0].id == 1

    mask = index.compile_filter({3, 4, 5})
    assert index.search(q, 1, admissible=mask)[0].id == 3


def test_filtered_search_equals_search_over_the_admissible_subset(
    index: BruteForceIndex,
) -> None:
    """Two independent routes to the same answer.

    The filtered oracle must agree with an index BUILT only from admissible
    rows. Every recall number today is measured against this, so if the two
    disagree the entire day's results are meaningless.
    """
    admissible = {2, 4, 5}
    subset = BruteForceIndex()
    subset.build(
        np.array([2, 4, 5], dtype=np.int64),
        np.stack([index._vectors[1], index._vectors[3], index._vectors[4]]),
    )

    q = unit(np.array([0.7, 0.5, 0.1]))
    filtered = index.search(q, 3, admissible=index.compile_filter(admissible))
    direct = subset.search(q, 3)

    assert [h.id for h in filtered] == [h.id for h in direct]
    assert [h.distance for h in filtered] == pytest.approx([h.distance for h in direct], abs=1e-6)


def test_k_is_clamped_to_the_admissible_count(index: BruteForceIndex) -> None:
    """Never pad results with rows that failed the filter."""
    hits = index.search(unit(np.array([1.0, 0.0, 0.0])), 10, admissible=index.compile_filter({4}))
    assert [h.id for h in hits] == [4]


def test_empty_admissible_set_returns_nothing(index: BruteForceIndex) -> None:
    hits = index.search(unit(np.array([1.0, 0.0, 0.0])), 5, admissible=index.compile_filter(set()))
    assert hits == []


def test_no_result_ever_has_infinite_distance(index: BruteForceIndex) -> None:
    """+inf masking must never leak into output."""
    mask = index.compile_filter({5})
    for hit in index.search(unit(np.array([1.0, 0.0, 0.0])), 5, admissible=mask):
        assert np.isfinite(hit.distance)


def test_selectivity_arithmetic() -> None:
    adm = AdmissibleSet(ids=frozenset({1, 2, 3}), label="t", corpus_size=300)
    assert adm.size == 3
    assert adm.selectivity == pytest.approx(0.01)


def test_selectivity_of_empty_corpus_is_zero() -> None:
    assert AdmissibleSet(ids=frozenset(), label="t", corpus_size=0).selectivity == 0.0


# --------------------------------------------------------------- against the DB


@pytest.mark.integration
async def test_admissible_ids_match_the_temporal_predicate(clean_db: Database) -> None:
    """The id set and the as-of read must agree — same predicate, same rows."""
    async with clean_db.acquire() as conn, conn.transaction():
        store = CorpusStore(conn)
        src = await store.upsert_source("ecfr", "eCFR", "u")
        doc = await store.upsert_document(src, "title-8/part-214", "T", "8 CFR Part 214")

        async def add(section: str, frm: date, to: date | None, text: str) -> None:
            digest = sha256(text.encode()).digest()
            sv = await conn.fetchval(
                """INSERT INTO section_versions
                     (document_id, section, effective_from, effective_to,
                      issue_date, content_sha256)
                   VALUES ($1,$2,$3,$4,$3,$5) RETURNING id""",
                doc,
                section,
                frm,
                to,
                digest,
            )
            await conn.execute(
                """INSERT INTO chunks (section_version_id, ordinal, citation_path,
                                       text, char_count, token_count, content_sha256)
                   VALUES ($1,0,$2,$3,$4,$5,$6)""",
                sv,
                f"8 CFR {section}(a)",
                text,
                len(text),
                2,
                digest,
            )

        await add("214.2", date(2017, 1, 18), date(2020, 10, 2), "OLD")
        await add("214.2", date(2020, 10, 2), None, "NEW")

        early = AsOf(valid_on=date(2019, 6, 1))
        late = AsOf(valid_on=date(2024, 1, 1))

        ids_early = await store.admissible_chunk_ids(doc, early)
        ids_late = await store.admissible_chunk_ids(doc, late)

        assert len(ids_early) == 1
        assert len(ids_late) == 1
        assert ids_early != ids_late

        # And they agree with the as-of text read.
        texts = {c.text for c in await store.chunks_as_of(doc, early)}
        assert texts == {"OLD"}
