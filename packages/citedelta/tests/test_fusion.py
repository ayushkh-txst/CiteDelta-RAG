"""RRF: the arithmetic, and the invariants that keep it deterministic."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from citedelta.fusion import RRF_K, RankedList, reciprocal_rank_fusion


def test_score_is_the_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([RankedList("a", (7, 8)), RankedList("b", (8, 7))])
    expected = 1 / (RRF_K + 1) + 1 / (RRF_K + 2)
    assert fused[0].score == pytest.approx(expected)
    assert fused[1].score == pytest.approx(expected)
    assert [h.chunk_id for h in fused] == [7, 8]  # tie broken by id


def test_consensus_beats_a_single_first_place() -> None:
    """The reason k=60 exists. Second in both lists must beat first in one."""
    fused = reciprocal_rank_fusion([RankedList("a", (1, 2)), RankedList("b", (3, 2))])
    assert fused[0].chunk_id == 2


def test_without_damping_a_single_first_place_would_win() -> None:
    """Contrast: at k=0 the fusion degenerates to 'most confident wins'."""
    fused = reciprocal_rank_fusion([RankedList("a", (1, 2)), RankedList("b", (3, 2))], k=0)
    assert fused[0].chunk_id in (1, 3)


def test_ranks_are_recorded_for_the_trace() -> None:
    fused = reciprocal_rank_fusion([RankedList("lexical", (5, 6)), RankedList("vector", (6,))])
    by_id = {h.chunk_id: h.ranks for h in fused}
    assert by_id[6] == {"lexical": 2, "vector": 1}
    assert by_id[5] == {"lexical": 1}


def test_duplicate_id_in_one_list_is_not_double_counted() -> None:
    """A retriever returning the same id twice must not buy a higher score."""
    once = reciprocal_rank_fusion([RankedList("a", (4,))])
    twice = reciprocal_rank_fusion([RankedList("a", (4, 4))])
    assert twice[0].score == pytest.approx(once[0].score)
    assert twice[0].ranks == {"a": 1}


def test_empty_and_single_list_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([RankedList("a", ())]) == []
    assert [h.chunk_id for h in reciprocal_rank_fusion([RankedList("a", (9, 8))])] == [9, 8]


def test_limit_truncates_after_ranking_not_before() -> None:
    lists = [RankedList("a", (1, 2, 3)), RankedList("b", (3, 2, 1))]
    assert [h.chunk_id for h in reciprocal_rank_fusion(lists, limit=2)] == [
        h.chunk_id for h in reciprocal_rank_fusion(lists)
    ][:2]


# ------------------------------------------------------------------ properties


@st.composite
def ranked_lists(draw: st.DrawFn) -> list[RankedList]:
    n_lists = draw(st.integers(min_value=1, max_value=4))
    universe = draw(st.lists(st.integers(1, 60), min_size=1, max_size=25, unique=True))
    out = []
    for i in range(n_lists):
        picked = draw(st.lists(st.sampled_from(universe), max_size=len(universe), unique=True))
        out.append(RankedList(f"r{i}", tuple(picked)))
    return out


@given(lists=ranked_lists())
def test_fusion_is_invariant_to_the_order_lists_are_passed(lists: list[RankedList]) -> None:
    """The property the spec singles out: 'RRF is invariant to input order
    given the same rank sets.'

    A dict-insertion-ordered tie-break would pass every hand-written example
    above and fail here on near-ties. This is the test that earns the
    determinism claim.
    """
    forward = reciprocal_rank_fusion(lists)
    backward = reciprocal_rank_fusion(list(reversed(lists)))
    assert [h.chunk_id for h in forward] == [h.chunk_id for h in backward]
    assert [round(h.score, 12) for h in forward] == [round(h.score, 12) for h in backward]


@given(lists=ranked_lists())
def test_output_is_sorted_by_descending_score(lists: list[RankedList]) -> None:
    fused = reciprocal_rank_fusion(lists)
    assert [h.score for h in fused] == sorted((h.score for h in fused), reverse=True)


@given(lists=ranked_lists())
def test_fusion_invents_nothing_and_loses_nothing(lists: list[RankedList]) -> None:
    given_ids = {i for ranked in lists for i in ranked.ids}
    assert {h.chunk_id for h in reciprocal_rank_fusion(lists)} == given_ids


@given(lists=ranked_lists())
def test_every_score_is_positive_and_bounded(lists: list[RankedList]) -> None:
    for hit in reciprocal_rank_fusion(lists):
        assert 0.0 < hit.score <= len(lists) / (RRF_K + 1)
