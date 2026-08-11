"""Proving every citation before the user sees it.

The strong claim this project makes is: 'every citation is real, was actually
retrieved, and was in force on the date you asked about.' This module is that
claim. Everything else is retrieval quality; this is the guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from citedelta.temporal import AdmissibleSet


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
    citation_ids: list[int],
    *,
    retrieved_ids: set[int],
    admissible: AdmissibleSet,
    corpus_ids: set[int] | None = None,
) -> ValidationResult:
    """Three independent checks per cited id.

    They are not equally likely to fire, and that is the point — each one
    catches a different party's mistake:

      exists      -> catches the MODEL inventing an id out of nothing.
      retrieved   -> catches the MODEL citing something real that it was
                     never shown. (Also implies `exists`, given a sane
                     retriever — kept separate so the failure message says
                     which thing went wrong.)
      admissible  -> catches US. Retrieval already filtered by as_of, so a
                     retrieved id is admissible BY CONSTRUCTION. This check
                     re-asserts that invariant at the boundary where it
                     becomes a user-visible promise, independently of the
                     code that was supposed to enforce it.

    That last one deserves defending, because "it can never fire" is usually
    an argument for deleting a check. It stays because the invariant it
    guards is the product's headline claim, the assertion costs a set
    lookup, and if it ever DOES fire it means the temporal filter has a bug
    that would otherwise reach a user as a confidently wrong answer about
    what the law said. That is precisely the class of failure worth paying a
    redundant check for.
    """
    failures: list[ValidationFailure] = []
    seen: list[int] = []

    for chunk_id in citation_ids:
        if chunk_id in seen:
            continue
        seen.append(chunk_id)

        if corpus_ids is not None and chunk_id not in corpus_ids:
            failures.append(
                ValidationFailure(chunk_id, "exists", f"chunk {chunk_id} is not in the corpus")
            )
            continue

        if chunk_id not in retrieved_ids:
            failures.append(
                ValidationFailure(
                    chunk_id,
                    "retrieved",
                    f"chunk {chunk_id} was never shown to the model",
                )
            )
            continue

        if chunk_id not in admissible.ids:
            failures.append(
                ValidationFailure(
                    chunk_id,
                    "admissible",
                    f"chunk {chunk_id} was not in force at {admissible.label}",
                )
            )

    return ValidationResult(cited=tuple(seen), failures=tuple(failures))
