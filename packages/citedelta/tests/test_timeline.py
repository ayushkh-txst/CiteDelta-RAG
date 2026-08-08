"""Timeline construction, against the quirks in the real source data."""

from __future__ import annotations

from datetime import date

from citedelta.ecfr.models import VersionRecord
from citedelta.ecfr.timeline import build_timelines, interval_covering


def _rec(section: str, eff: str, issued: str, *, removed: bool = False) -> VersionRecord:
    return VersionRecord.model_validate(
        {
            "identifier": section,
            "date": eff,
            "issue_date": issued,
            "removed": removed,
            "substantive": True,
            "name": "",
        }
    )


def test_consecutive_versions_close_each_other() -> None:
    tl = build_timelines(
        [
            _rec("214.2", "2017-01-18", "2017-01-18"),
            _rec("214.2", "2020-10-02", "2020-10-02"),
        ]
    )["214.2"]
    assert tl[0].effective_from == date(2017, 1, 18)
    assert tl[0].effective_to == date(2020, 10, 2)
    assert tl[1].effective_to is None  # still in force


def test_duplicate_valid_dates_collapse_keeping_earliest_issue() -> None:
    """28 such duplicates exist in part 214. Without this the partial unique
    index in Block 1 rejects the second insert and ingestion dies."""
    tl = build_timelines(
        [
            _rec("214.11", "2020-10-02", "2022-01-05"),
            _rec("214.11", "2020-10-02", "2020-10-02"),
            _rec("214.11", "2020-10-02", "2021-06-30"),
        ]
    )["214.11"]
    assert len(tl) == 1
    assert tl[0].issue_date == date(2020, 10, 2)


def test_removed_sections_keep_an_interval() -> None:
    tl = build_timelines(
        [
            _rec("214.16", "2016-12-23", "2016-12-23"),
            _rec("214.16", "2019-05-10", "2019-05-10", removed=True),
        ]
    )["214.16"]
    assert tl[1].removed is True


def test_interval_boundaries_are_half_open() -> None:
    tl = build_timelines(
        [
            _rec("214.2", "2017-01-18", "2017-01-18"),
            _rec("214.2", "2020-10-02", "2020-10-02"),
        ]
    )
    day_before = interval_covering(tl, "214.2", date(2020, 10, 1))
    assert day_before is not None
    assert day_before.effective_from == date(2017, 1, 18)

    change_day = interval_covering(tl, "214.2", date(2020, 10, 2))
    assert change_day is not None
    assert change_day.effective_from == date(2020, 10, 2)

    assert interval_covering(tl, "214.2", date(2015, 1, 1)) is None
