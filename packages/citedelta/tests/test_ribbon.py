from __future__ import annotations

from datetime import date

from citedelta.answer.models import Citation
from citedelta.web.ribbon import build_ribbon


def _c(path: str, start: str, end: str | None) -> Citation:
    return Citation(
        chunk_id=1,
        citation_path=path,
        effective_from=start,
        effective_to=end,
        text="t",
        rrf_score=0.03,
    )


def test_open_ended_provision_is_in_force() -> None:
    r = build_ribbon([_c("8 CFR 214.2(f)", "2016-04-01", None)], as_of=date(2026, 8, 11))
    assert r.bars[0].in_force
    assert not r.has_superseded


def test_provision_superseded_before_as_of_is_marked_gone() -> None:
    r = build_ribbon([_c("8 CFR 214.2(f)", "2016-04-01", "2019-01-01")], as_of=date(2026, 8, 11))
    assert not r.bars[0].in_force
    assert r.has_superseded


def test_provision_not_yet_effective_at_as_of_is_not_in_force() -> None:
    r = build_ribbon([_c("8 CFR 214.2(f)", "2021-01-20", None)], as_of=date(2019, 6, 1))
    assert not r.bars[0].in_force


def test_effective_to_is_exclusive_at_the_boundary() -> None:
    """Same convention as section_versions and the rate table: the end date
    belongs to the successor, so a provision is NOT in force on its own
    effective_to."""
    on = build_ribbon([_c("x", "2016-01-01", "2020-01-01")], as_of=date(2019, 12, 31))
    off = build_ribbon([_c("x", "2016-01-01", "2020-01-01")], as_of=date(2020, 1, 1))
    assert on.bars[0].in_force
    assert not off.bars[0].in_force


def test_bars_stay_inside_the_plot_area() -> None:
    """A range extending past the window must clamp, not overflow the SVG."""
    r = build_ribbon([_c("x", "1990-01-01", "2099-01-01")], as_of=date(2026, 8, 11))
    bar = r.bars[0]
    assert bar.x >= r.axis_x0
    assert bar.x + bar.width <= r.axis_x1 + 0.001


def test_a_same_day_range_is_still_visible() -> None:
    r = build_ribbon([_c("x", "2020-01-01", "2020-01-02")], as_of=date(2026, 8, 11))
    assert r.bars[0].width >= 3.0


def test_height_grows_with_the_number_of_bars() -> None:
    one = build_ribbon([_c("x", "2016-01-01", None)], as_of=date(2026, 8, 11))
    three = build_ribbon([_c("x", "2016-01-01", None)] * 3, as_of=date(2026, 8, 11))
    assert three.height > one.height


def test_marker_sits_inside_the_axis() -> None:
    r = build_ribbon([_c("x", "2016-01-01", None)], as_of=date(2026, 8, 11))
    assert r.axis_x0 < r.marker_x < r.axis_x1
