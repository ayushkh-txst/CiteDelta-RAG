"""Proving every citation before the user sees it.

The strong claim this project makes is: 'every citation is real, was actually
retrieved, and was in force on the date you asked about.' This module is that
claim. Everything else is retrieval quality; this is the guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from citedelta.answer.models import Citation
from citedelta.temporal import AdmissibleSet


@dataclass(frozen=True, slots=True)
class CitedRef:
    """What the model claims: an id and the words it says are there."""

    chunk_id: int
    quote: str


def normalize(text: str) -> str:
    """Collapse all whitespace to single spaces.

    Regulation text arrives with line breaks and runs of spaces from the XML;
    a model copying a span faithfully will reflow it. Whitespace is the one
    difference that is never semantic, so it is the one we forgive.

    Case is NOT normalized. "Student" and "student" are a real difference in
    a legal quotation, and forgiving case is the first step down a slope that
    ends with forgiving paraphrase — at which point the check means nothing.
    """
    return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    chunk_id: int
    check: str
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    cited: tuple[int, ...]
    failures: tuple[ValidationFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_citations(
    refs: list[CitedRef],
    *,
    retrieved: Mapping[int, Citation],
    admissible: AdmissibleSet,
    corpus_ids: set[int] | None = None,
) -> ValidationResult:
    """Four independent checks per cited id.

      exists      -> the MODEL invented an id out of nothing
      retrieved   -> the MODEL cited something real it was never shown
      admissible  -> WE have a bug; retrieval already filtered by as_of, so
                     this re-asserts the product's headline invariant at the
                     boundary where it becomes a user-visible promise
      quote       -> the MODEL paraphrased while claiming to quote

    `quote` is the new one and it is the strongest of the four, because it is
    the only check on the RELATIONSHIP between the answer and the source. The
    other three prove a citation points somewhere legitimate; this one proves
    the words the answer leans on are actually there.
    """
    failures: list[ValidationFailure] = []
    seen: list[int] = []

    for ref in refs:
        if ref.chunk_id in seen:
            continue
        seen.append(ref.chunk_id)

        if corpus_ids is not None and ref.chunk_id not in corpus_ids:
            failures.append(
                ValidationFailure(
                    ref.chunk_id, "exists", f"chunk {ref.chunk_id} is not in the corpus"
                )
            )
            continue

        source = retrieved.get(ref.chunk_id)
        if source is None:
            failures.append(
                ValidationFailure(
                    ref.chunk_id,
                    "retrieved",
                    f"chunk {ref.chunk_id} was never shown to the model",
                )
            )
            continue

        if ref.chunk_id not in admissible.ids:
            failures.append(
                ValidationFailure(
                    ref.chunk_id,
                    "admissible",
                    f"chunk {ref.chunk_id} was not in force at {admissible.label}",
                )
            )
            continue

        if not ref.quote.strip():
            failures.append(
                ValidationFailure(
                    ref.chunk_id, "quote", f"chunk {ref.chunk_id} was cited without a quote"
                )
            )
            continue

        if normalize(ref.quote) not in normalize(source.text):
            failures.append(
                ValidationFailure(
                    ref.chunk_id, "quote", f"quoted text does not appear in chunk {ref.chunk_id}"
                )
            )

    return ValidationResult(cited=tuple(seen), failures=tuple(failures))
