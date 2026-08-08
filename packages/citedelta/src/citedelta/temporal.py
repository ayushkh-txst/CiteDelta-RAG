"""Value objects for asking temporal questions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime


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
