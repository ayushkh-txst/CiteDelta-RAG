"""Turn a thread of results into a renderable record.

Pure arithmetic over dates. No template concerns, no HTML — which is what
makes the rupture logic testable rather than something you verify by
squinting at a browser.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date

from citedelta.answer.models import AnswerResult


@dataclass(frozen=True, slots=True)
class TurnView:
    question: str
    as_of: date
    result: AnswerResult

    resolved_query: str | None
    """Set only when it differs from `question` — i.e. this was a follow-up
    that got rewritten. The template shows it as `↳ searched: …` because
    otherwise the record misrepresents what happened: the user typed "what
    about then?" and we searched for something else."""

    continuation: bool
    """True when the previous turn had the same as_of. The gutter then draws
    a hairline instead of repeating the date, so a CHANGE is the loud thing
    rather than one date among many identical ones."""

    kind: str = "turn"

    @property
    def stamp_day(self) -> str:
        return f"{self.as_of.day} {self.as_of:%b}"

    @property
    def stamp_year(self) -> str:
        return f"{self.as_of:%Y}"


@dataclass(frozen=True, slots=True)
class Rupture:
    """Inserted between two turns whose as_of differs."""

    as_of: date
    amendments_between: int
    earlier: bool

    kind: str = "rupture"

    @property
    def label(self) -> str:
        return f"{self.as_of.day} {self.as_of:%B} {self.as_of:%Y}"

    @property
    def glyph(self) -> str:
        return "⤺" if self.earlier else "⤻"

    @property
    def count_label(self) -> str:
        """Zero is a real and interesting outcome: the user moved the date but
        landed inside the same amendment window, so the law did not change and
        the answer will not either. Saying so prevents them reading an
        identical answer as a bug."""
        if self.amendments_between == 0:
            return "· no amendments in between"
        word = "amendment" if self.amendments_between == 1 else "amendments"
        return f"· {self.amendments_between} {word} {'earlier' if self.earlier else 'later'}"


def amendments_between(a: date, b: date, amendment_dates: list[date]) -> int:
    """How many amendment dates lie strictly between two points.

    `amendment_dates` is the sorted list of real effective dates loaded at
    startup, so this is a bisect rather than a scan. Strictly between, because
    an amendment landing exactly on the boundary is the law you are now
    looking at, not something you skipped past.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    return bisect.bisect_left(amendment_dates, hi) - bisect.bisect_right(amendment_dates, lo)


def build_transcript(
    turns: list[TurnView], *, amendment_dates: list[date]
) -> list[TurnView | Rupture]:
    """Interleave ruptures into the turn sequence."""
    out: list[TurnView | Rupture] = []
    previous: date | None = None
    for turn in turns:
        if previous is not None and turn.as_of != previous:
            out.append(
                Rupture(
                    as_of=turn.as_of,
                    amendments_between=amendments_between(previous, turn.as_of, amendment_dates),
                    earlier=turn.as_of < previous,
                )
            )
        out.append(turn)
        previous = turn.as_of
    return out
