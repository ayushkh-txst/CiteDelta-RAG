"""What the answer path can produce. There are exactly two outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Literal

from citedelta.retrieve import RetrievalTrace


class RefusalReason(StrEnum):
    """Why we didn't answer.

    Every value here is a SUCCESS path. The one thing this enum must never
    grow is a catch-all like `ERROR` — an exception is not a refusal, and
    collapsing the two would let a crashed retriever render as a thoughtful
    decision not to answer.
    """

    GREETING = "greeting"
    """Not a regulation question. Recorded as a refusal because no question
    was answered — which is accurate — but RENDERED as a conversation, not
    as a refusal card. See web/copy.py and the turn partial.

    Keeping this inside RefusalReason rather than adding a third result type
    is deliberate: `query_traces` has a CHECK constraint asserting a row is
    an answer XOR a refusal, and persist() plus every reader depends on
    exactly two outcomes. A third kind would ripple through the schema, the
    constraint, and the trace API to express something the UI can express
    with one branch."""

    OUT_OF_SCOPE = "out_of_scope"
    """A real question, but not about this corpus. Set by the model, not by
    a retrieval threshold — measured, no scoring gate can separate 'out of
    scope' from 'not relevant'."""

    NO_ADMISSIBLE_SOURCE = "no_admissible_source"
    """Nothing in the corpus was in force at as_of that matched. Common and
    correct when time-travelling to before a provision existed."""

    LOW_CONFIDENCE = "low_confidence"
    """Retrieval found things, but nothing scored well enough to be worth
    generating from."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    """The model read the admissible text and said it does not answer the
    question. Distinct from LOW_CONFIDENCE: retrieval was confident, the
    source genuinely doesn't cover it."""

    FABRICATED_CITATION = "fabricated_citation"
    """The validator rejected a cited id. The answer text is DISCARDED, not
    shown with a warning — see validator.py for why."""

    PROVIDER_REFUSED = "provider_refused"
    """The provider's own safety classifier declined. HTTP 200, not an error."""

    MALFORMED_RESPONSE = "malformed_response"
    """Structured output came back unusable. Rare enough to be worth its own
    code rather than being folded into a generic failure."""


@dataclass(frozen=True, slots=True)
class Citation:
    """One cited chunk, with everything the UI needs to render it."""

    chunk_id: int
    citation_path: str
    effective_from: str
    effective_to: str | None
    text: str
    rrf_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    """retriever name -> 1-based rank. `{"vector": 2}` with no "lexical" key
    means BM25 never returned this at all — which is exactly the kind of thing
    the trace panel exists to show."""

    @property
    def in_force_label(self) -> str:
        end = self.effective_to or "present"
        return f"{self.effective_from} → {end}"


@dataclass(frozen=True, slots=True)
class Answer:
    query: str
    as_of: str
    text: str
    citations: tuple[Citation, ...]
    trace: RetrievalTrace
    cost_usd: Decimal
    latency_ms: float

    refused: ClassVar[Literal[False]] = False


@dataclass(frozen=True, slots=True)
class Refusal:
    query: str
    as_of: str
    reason: RefusalReason
    detail: str
    """Human-readable, shown in the UI. Never a stack trace."""

    trace: RetrievalTrace | None
    cost_usd: Decimal
    latency_ms: float

    refused: ClassVar[Literal[True]] = True


AnswerResult = Answer | Refusal
