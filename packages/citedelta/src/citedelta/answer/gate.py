"""Pre-flight checks. Cheap, and they run before any tokens are spent."""

from __future__ import annotations

from dataclasses import dataclass

from citedelta.answer.models import RefusalReason
from citedelta.retrieve import RetrievalTrace

MIN_RRF_SCORE = 0.02
"""Below this, the best fused hit is a chunk that one retriever ranked poorly
and the other didn't return at all.

Calibrate it, don't guess it: RRF scores are 1/(60+rank) summed across
retrievers, so a single retriever's rank-1 hit scores 1/61 = 0.0164 and
agreement at rank 1 from both scores 0.0328. 0.02 therefore means roughly
'at least one retriever was confident, or both were mildly positive'. The
60-case eval re-tunes this later; treat it as provisional and say so in the
ADR."""


@dataclass(frozen=True, slots=True)
class GateVerdict:
    passed: bool
    reason: RefusalReason | None = None
    detail: str = ""


def pre_flight(trace: RetrievalTrace) -> GateVerdict:
    if not trace.hits:
        return GateVerdict(
            False,
            RefusalReason.NO_ADMISSIBLE_SOURCE,
            f"No provision in force on {trace.as_of} matched this question.",
        )

    top = trace.hits[0].score
    if top < MIN_RRF_SCORE:
        return GateVerdict(
            False,
            RefusalReason.LOW_CONFIDENCE,
            ("The closest matching provisions were only weakly related to the question."),
        )

    return GateVerdict(True)
