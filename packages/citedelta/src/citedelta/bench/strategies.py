"""The three ways to combine a filter with a vector index."""

from __future__ import annotations

from dataclasses import dataclass

from citedelta.index.vector import BoolMask, Neighbor, VectorIndex, Vectors
from citedelta.temporal import AdmissibleSet


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    description: str


POST_FILTER = Strategy("post-filter", "retrieve k unfiltered, then discard inadmissible")
POST_FILTER_OVERFETCH = Strategy(
    "post-filter+overfetch", "retrieve k/selectivity unfiltered, then discard"
)
IN_INDEX = Strategy("in-index", "the filter is enforced during traversal")


def post_filter_search(
    index: VectorIndex,
    query: Vectors,
    k: int,
    admissible: AdmissibleSet,
    *,
    effort: int | None = None,
    overfetch: int = 1,
) -> list[Neighbor]:
    """Retrieve `k * overfetch` unfiltered, discard inadmissible, keep k.

    `overfetch=1` is the naive approach almost every RAG pipeline ships:
    top-k from the vector store, then a WHERE clause in application code.
    On a 1.9%-selectivity filter it returns nothing 81% of the time.

    Setting overfetch to ceil(1/selectivity) is the honest repair, and its
    cost is the entire point of the sweep: you are now scanning a percentage
    of the corpus proportional to 1/selectivity, which is what the index
    existed to avoid.
    """
    raw = index.search(query, max(k * overfetch, k), effort=effort)
    return [n for n in raw if n.id in admissible.ids][:k]


def in_index_search(
    index: VectorIndex,
    query: Vectors,
    k: int,
    mask: BoolMask,
    *,
    effort: int | None = None,
) -> list[Neighbor]:
    """The filter is pushed down into the index's own traversal."""
    return index.search(query, k, effort=effort, admissible=mask)


def required_overfetch(selectivity: float, *, cap: int = 512) -> int:
    """ceil(1/selectivity), the expected fetch needed for k admissible hits.

    Expectation only. The variance is what bites: at 1.9% selectivity you
    need ~525x on average, and recall doesn't actually reach 1.000 until
    ~250x k (2,500 candidates for k=10) because the admissible rows are not
    uniformly distributed among the nearest neighbours — all versions of one
    paragraph sit together in vector space, and exactly one of them is in
    force.
    """
    if selectivity <= 0.0:
        return cap
    return min(cap, max(1, round(1.0 / selectivity)))
