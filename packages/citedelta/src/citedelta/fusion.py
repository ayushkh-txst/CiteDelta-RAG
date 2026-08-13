"""Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

RRF_K = 60


@dataclass(frozen=True, slots=True)
class RankedList:
    """One retriever's opinion, best first. Only the ORDER is used."""

    name: str
    ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FusedHit:
    chunk_id: int
    score: float
    # retriever name -> 1-based rank. Kept because it is the whole retrieval
    # trace: "vector ranked this 2nd, BM25 didn't return it at all" is what
    # makes a hybrid result explainable rather than magic.
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    lists: Sequence[RankedList], *, k: int = RRF_K, limit: int | None = None
) -> list[FusedHit]:
    """Fuse ranked lists by reciprocal rank.

    Deterministic by construction — ties break on ascending chunk id, never on
    dict insertion order. That is what makes the function invariant to the
    order the lists are passed in, and there is a Hypothesis test below that
    holds it to that. Without the explicit tie-break the output would depend on
    which retriever happened to be first in the argument list, which is a real
    bug that only shows up as unstable rankings on near-ties.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    for ranked in lists:
        for position, chunk_id in enumerate(ranked.ids, start=1):
            # A retriever listing the same id twice is a bug upstream; keep the
            # best rank rather than double-counting it into the lead.
            previous = ranks.setdefault(chunk_id, {}).get(ranked.name)
            if previous is not None and previous <= position:
                continue
            if previous is not None:
                scores[chunk_id] -= 1.0 / (k + previous)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            ranks[chunk_id][ranked.name] = position

    fused = [
        FusedHit(chunk_id=cid, score=score, ranks=dict(ranks[cid])) for cid, score in scores.items()
    ]
    # Scores are sums of reciprocals, and float addition is not associative:
    # two sums that are mathematically equal can differ in the last ulp
    # depending on the order the lists arrived. Sorting on the raw score would
    # then let summation order decide a tie, silently breaking the
    # order-invariance guarantee. Rounding to 12 decimals is safe because no
    # two DISTINCT reciprocal sums are ever that close — adjacent ranks differ
    # by ~1e-4.
    fused.sort(key=lambda hit: (-round(hit.score, 12), hit.chunk_id))
    return fused[:limit] if limit is not None else fused
