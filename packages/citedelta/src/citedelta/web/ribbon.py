"""Server-computed SVG for the temporal ribbon.

Everything here is date arithmetic. There is no chart library because there is
no chart — there are four rectangles whose x-positions are a linear map from a
date range onto a pixel range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from citedelta.answer.models import Citation

# Geometry. The label gutter is fixed because citation paths are monospace,
# so their rendered width is predictable in a way proportional text is not.
GUTTER = 150
RIGHT_PAD = 38
WIDTH = 860
ROW_H = 32
BAR_H = 14
TOP = 20


@dataclass(frozen=True, slots=True)
class RibbonBar:
    label: str
    x: float
    width: float
    in_force: bool
    title: str


@dataclass(frozen=True, slots=True)
class Ribbon:
    bars: tuple[RibbonBar, ...]
    ticks: tuple[tuple[float, str], ...]
    marker_x: float
    marker_label: str
    height: int
    axis_x0: float
    axis_x1: float

    @property
    def has_superseded(self) -> bool:
        return any(not b.in_force for b in self.bars)


def build_ribbon(
    citations: list[Citation],
    *,
    as_of: date,
    span_start: date | None = None,
    span_end: date | None = None,
) -> Ribbon:
    """Map effective ranges onto pixels.

    The window is padded past `as_of` on the right so the marker never lands
    flush on the axis end — a marker at the exact edge reads as "the data
    stops here" rather than "you are here".
    """
    start = span_start or date(2016, 1, 1)
    end = span_end or date(max(as_of.year + 1, start.year + 2), 1, 1)
    total_days = (end - start).days or 1
    plot_w = WIDTH - GUTTER - RIGHT_PAD

    def x_for(d: date) -> float:
        clamped = min(max(d, start), end)
        return GUTTER + plot_w * ((clamped - start).days / total_days)

    bars: list[RibbonBar] = []
    for c in citations:
        eff_from = date.fromisoformat(c.effective_from)
        eff_to = date.fromisoformat(c.effective_to) if c.effective_to else end
        x0, x1 = x_for(eff_from), x_for(eff_to)
        # An effective range shorter than the axis resolution would render as
        # an invisible zero-width rect. Floor it so a same-day supersession is
        # still something you can see.
        width = max(x1 - x0, 3.0)
        in_force = eff_from <= as_of and (c.effective_to is None or as_of < eff_to)
        bars.append(
            RibbonBar(
                label=c.citation_path.replace("8 CFR ", ""),
                x=x0,
                width=width,
                in_force=in_force,
                title=f"{c.citation_path}: {c.in_force_label}",
            )
        )

    step = 2 if (end.year - start.year) > 6 else 1
    ticks = tuple((x_for(date(y, 1, 1)), str(y)) for y in range(start.year, end.year + 1, step))

    return Ribbon(
        bars=tuple(bars),
        ticks=ticks,
        marker_x=x_for(as_of),
        marker_label=f"as of {as_of.strftime('%-d %b %Y')}",
        height=TOP + len(bars) * ROW_H + 26,
        axis_x0=GUTTER,
        axis_x1=WIDTH - RIGHT_PAD,
    )
