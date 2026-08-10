"""The hybrid retrieval path: both retrievers, one temporal predicate."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from citedelta.fusion import FusedHit, RankedList, reciprocal_rank_fusion
from citedelta.index.lexical import LexicalIndex
from citedelta.index.vector import VectorIndex, Vectors
from citedelta.temporal import AdmissibleSet

log = structlog.get_logger(__name__)


@dataclass
class RetrievalTrace:
    """Everything that happened, for the inspector panel.

    Persisting the trace is why the UI can show *why* a chunk was returned.
    Recording it at retrieval time costs nothing; reconstructing it afterwards
    is impossible.
    """

    query: str
    as_of: str
    selectivity: float
    candidates_lexical: int = 0
    candidates_vector: int = 0
    fused: int = 0
    hits: list[FusedHit] = field(default_factory=list)


def hybrid_search(
    query: str,
    query_vector: Vectors,
    *,
    lexical: LexicalIndex,
    vector: VectorIndex,
    admissible: AdmissibleSet,
    k: int = 10,
    candidates_per_retriever: int = 50,
    effort: int | None = None,
) -> RetrievalTrace:
    """Retrieve k admissible chunks using both retrievers.

    Both sides are filtered by the SAME admissible set, pushed down into each
    index. Filtering only one side would be worse than filtering neither: the
    fusion would systematically favour whichever retriever was allowed to
    return superseded text, because it would be the only one supplying
    candidates for half the queries.

    `candidates_per_retriever` is deliberately larger than k. Fusion needs
    depth to find agreement — with 10 from each you frequently get two
    disjoint sets of 10 and no consensus to detect.
    """
    lexical_mask = lexical.compile_filter(admissible.ids)
    vector_mask = vector.compile_filter(admissible.ids)

    lexical_hits = lexical.search(query, candidates_per_retriever, admissible=lexical_mask)
    vector_hits = vector.search(
        query_vector, candidates_per_retriever, effort=effort, admissible=vector_mask
    )

    fused = reciprocal_rank_fusion(
        [
            RankedList("lexical", tuple(h.chunk_id for h in lexical_hits)),
            RankedList("vector", tuple(h.id for h in vector_hits)),
        ],
        limit=k,
    )

    trace = RetrievalTrace(
        query=query,
        as_of=admissible.label,
        selectivity=admissible.selectivity,
        candidates_lexical=len(lexical_hits),
        candidates_vector=len(vector_hits),
        fused=len(fused),
        hits=fused,
    )
    log.info(
        "retrieve.hybrid",
        query=query[:60],
        lexical=trace.candidates_lexical,
        vector=trace.candidates_vector,
        fused=trace.fused,
    )
    return trace
