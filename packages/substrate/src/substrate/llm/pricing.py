"""What a call cost.

Rates change, and one of them changes within this project's own timeline.
That makes the price table a small bitemporal problem — which is a pleasant
coincidence, because bitemporal modelling is the thing this whole project is
about. Same shape as `section_versions`: a row is valid over a date range,
you look it up as of a point in time, and superseded rows are kept rather
than overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from substrate.llm import TokenUsage

# Multipliers applied to the INPUT rate. Cache writes cost more than fresh
# input; cache reads cost a tenth. These are the numbers that make caching
# visible in the ledger.
CACHE_WRITE_MULTIPLIER = Decimal("1.25")
CACHE_READ_MULTIPLIER = Decimal("0.1")

_PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class Rate:
    """Dollars per million tokens, valid over a date range.

    `effective_to` is exclusive and None means "still current" — exactly the
    convention `section_versions` uses, so there is one temporal idiom in the
    codebase rather than two.
    """

    model: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    effective_from: date
    effective_to: date | None = None

    def covers(self, when: date) -> bool:
        if when < self.effective_from:
            return False
        return self.effective_to is None or when < self.effective_to


# Verified against the provider's published rates on 2026-08-11.
# The Sonnet introductory rate expires 2026-08-31, which is INSIDE this
# project's own timeline. That is why these are rows with dates rather
# than constants.
RATES: tuple[Rate, ...] = (
    Rate("claude-opus-5", Decimal("5.00"), Decimal("25.00"), date(2026, 1, 1)),
    Rate(
        "claude-sonnet-5",
        Decimal("2.00"),
        Decimal("10.00"),
        date(2026, 1, 1),
        date(2026, 9, 1),
    ),
    Rate("claude-sonnet-5", Decimal("3.00"), Decimal("15.00"), date(2026, 9, 1)),
    Rate("claude-haiku-4-5", Decimal("1.00"), Decimal("5.00"), date(2026, 1, 1)),
)


class UnknownRate(LookupError):
    """No published rate for this model on this date.

    Raised rather than defaulted. A silent zero would make an unpriced model
    look free, and "free" is the one wrong answer that never prompts anyone
    to go looking.
    """


def rate_for(model: str, when: date) -> Rate:
    for rate in RATES:
        if rate.model == model and rate.covers(when):
            return rate
    raise UnknownRate(f"no rate for {model!r} as of {when.isoformat()}")


def price(usage: TokenUsage, *, model: str, when: date) -> Decimal:
    """Cost in USD. Decimal throughout — this is money, not a metric.

    Rounded to 6 decimal places at the end, not per-bucket: a single query
    can genuinely cost less than a hundredth of a cent, and rounding four
    sub-cent buckets to cents individually floors the whole thing to zero.
    """
    r = rate_for(model, when)
    fresh_in = Decimal(usage.input_tokens) * r.input_per_mtok
    cache_w = Decimal(usage.cache_write_tokens) * r.input_per_mtok * CACHE_WRITE_MULTIPLIER
    cache_r = Decimal(usage.cache_read_tokens) * r.input_per_mtok * CACHE_READ_MULTIPLIER
    out = Decimal(usage.output_tokens) * r.output_per_mtok
    total = (fresh_in + cache_w + cache_r + out) / _PER_MILLION
    return total.quantize(Decimal("0.000001"))


@dataclass
class CostLedger:
    """Running spend, keyed by run_id.

    The project budget sits under $80 across its components. Unmeasured spend
    is how that gets blown — not by one expensive call, but by a loop nobody
    was watching. This makes the number visible in the same log line as
    everything else.
    """

    _by_run: dict[str, Decimal] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._by_run = {}

    def record(self, run_id: str, cost: Decimal) -> Decimal:
        running = self._by_run.get(run_id, Decimal(0)) + cost
        self._by_run[run_id] = running
        return running

    def total(self, run_id: str | None = None) -> Decimal:
        if run_id is not None:
            return self._by_run.get(run_id, Decimal(0))
        return sum(self._by_run.values(), Decimal(0))
