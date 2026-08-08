"""Version records → validity intervals."""

from __future__ import annotations

from collections import defaultdict

from citedelta.ecfr.models import SectionInterval, VersionRecord


def build_timelines(records: list[VersionRecord]) -> dict[str, list[SectionInterval]]:
    """Group by section, dedupe on valid time, close each interval at the next.

    Two quirks in the real data:

    1. some (section, effective_from) pairs appear more than once — eCFR
       re-records the same amendment at a later issue_date. Valid time is what
       identifies a version, so collapse duplicates and keep the earliest
       issue_date, or the partial unique index rejects the insert.
    2. A record can be flagged `removed` — that's a tombstone, not an error.
       The section still gets an interval so 'in force in 2020?' returns
       nothing for it.
    """
    by_section: dict[str, dict[str, VersionRecord]] = defaultdict(dict)

    for rec in records:
        key = rec.effective_from.isoformat()
        existing = by_section[rec.section].get(key)
        if existing is None or rec.issue_date < existing.issue_date:
            by_section[rec.section][key] = rec

    timelines: dict[str, list[SectionInterval]] = {}
    for section, versions in by_section.items():
        ordered = sorted(versions.values(), key=lambda r: r.effective_from)
        intervals: list[SectionInterval] = []
        for i, rec in enumerate(ordered):
            nxt = ordered[i + 1].effective_from if i + 1 < len(ordered) else None
            intervals.append(
                SectionInterval(
                    section=section,
                    effective_from=rec.effective_from,
                    effective_to=nxt,
                    issue_date=rec.issue_date,
                    removed=rec.removed,
                )
            )
        timelines[section] = intervals
    return timelines


def interval_covering(
    timelines: dict[str, list[SectionInterval]], section: str, day: object
) -> SectionInterval | None:
    """The interval containing `day`, or None if the section wasn't tracked then."""
    from datetime import date as _date

    assert isinstance(day, _date)
    for iv in timelines.get(section, ()):
        if iv.contains(day):
            return iv
    return None
