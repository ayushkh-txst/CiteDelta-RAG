"""Typed shapes for everything crossing a module boundary."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class VersionRecord(BaseModel):
    """One row from /versions/title-N.json?part=M.

    effective_from — when the amendment took effect in the world (VALID time)
    issue_date     — when eCFR published it (TRANSACTION time)
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    section: str = Field(alias="identifier")  # '214.2'
    effective_from: date = Field(alias="date")
    issue_date: date
    name: str = ""
    removed: bool = False
    substantive: bool = True


class SectionInterval(BaseModel):
    """A section's validity window, derived from consecutive version records."""

    section: str
    effective_from: date
    effective_to: date | None  # None = still in force
    issue_date: date
    removed: bool = False

    def contains(self, day: date) -> bool:
        """Half-open [effective_from, effective_to), matching the SQL daterange."""
        if day < self.effective_from:
            return False
        return self.effective_to is None or day < self.effective_to


class ParsedChunk(BaseModel):
    """A retrievable unit of text with a citeable path."""

    ordinal: int
    citation_path: str  # '8 CFR 214.2(f)(10)(ii)(A)'
    text: str


class ParsedSection(BaseModel):
    section: str  # '214.2'
    heading: str
    chunks: list[ParsedChunk]
