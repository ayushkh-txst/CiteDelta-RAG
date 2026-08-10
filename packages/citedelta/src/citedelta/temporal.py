"""Value objects for asking temporal questions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Self


@dataclass(frozen=True, slots=True)
class AsOf:
    """A point in bitemporal space.

    valid_on — 'what was the rule in force on this date?'
    known_at — restricted to what had been recorded by this instant, or None
               to use everything we know now.
    """

    valid_on: date
    known_at: datetime | None = None

    @classmethod
    def today(cls) -> AsOf:
        return cls(valid_on=datetime.now(UTC).date())

    def __str__(self) -> str:
        if self.known_at is None:
            return f"valid_on={self.valid_on.isoformat()}"
        return f"valid_on={self.valid_on.isoformat()} known_at={self.known_at.isoformat()}"


@dataclass(frozen=True, slots=True)
class AdmissibleSet:
    """An admissible id set, carrying enough context to report on itself.

    `label` exists so results are self-describing: a benchmark row that says
    'selectivity 0.019' is far less useful six weeks later than one that says
    'as_of=2019-06-01'. Cheap to carry, and it ends up in the plot legend.
    """

    ids: frozenset[int]
    label: str
    corpus_size: int

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def selectivity(self) -> float:
        """Fraction of the corpus that survives the filter.

        This is THE number that determines whether post-filtering is viable.
        At 1.0 a post-filter is free; as it approaches zero the overfetch
        required to post-filter safely approaches the corpus size, at which
        point the index has stopped being an index.
        """
        return self.size / self.corpus_size if self.corpus_size else 0.0

    @classmethod
    def from_as_of(cls, ids: set[int], as_of: AsOf, corpus_size: int) -> Self:
        return cls(ids=frozenset(ids), label=str(as_of), corpus_size=corpus_size)
