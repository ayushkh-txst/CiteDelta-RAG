"""Jinja filters. The escaping here is load-bearing — read the docstrings."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, timedelta

from markupsafe import Markup, escape

from citedelta.answer.models import Citation

_CITE = re.compile(r"\[(\d+)\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MIN_QUOTE_TOKENS = 4
_STRENGTH = (("strongest", 0.92), ("strong", 0.75), ("moderate", 0.45), ("weak", 0.0))


def citation_chips(text: str, citations: list[Citation]) -> Markup:
    """Turn [37679] into a chip numbered by its position in *citations*.

    The model cites the opaque chunk ids WE assigned (see answer/prompt.py) —
    e.g. ``[37679]`` — but the sources card numbers its entries 1..n. This
    filter maps each id to that display ordinal, so ``[37679]`` renders as chip
    ``1`` linking to ``#cite-1``, matching the first source card.

    The escaping here matters, and the order is easy to get backwards:

        1. escape() the ENTIRE model output first
        2. THEN substitute chip markup into the escaped string

    Doing it the other way — substituting first, escaping after — would escape
    your own chip markup into visible angle brackets. Skipping the escape
    entirely would let model-generated text inject HTML into the page, and the
    model's output derives from regulation text a user can influence via the
    query. Escape first, always.
    """
    order = {c.chunk_id: i + 1 for i, c in enumerate(citations)}
    safe = str(escape(text))

    def chip(match: re.Match[str]) -> str:
        n = int(match.group(1))
        ordinal = order.get(n)
        if ordinal is None:
            return f"[{n}]"
        return f'<a class="chip" href="#cite-{ordinal}">{ordinal}</a>'

    return Markup(_CITE.sub(chip, safe))  # noqa: S704 - output is escaped first, see docstring


def markdown_lite(text: str) -> Markup:
    """Bold only. Escape first, then substitute — same order and same reason
    as citation_chips: this function marks its output safe, so it owns the
    escaping, and doing it the other way round would either double-escape the
    markup or open an injection path."""
    return Markup(_BOLD.sub(r"<strong>\1</strong>", str(escape(text))))  # noqa: S704


def ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 3 -> 3rd, 11 -> 11th. Plain English for ranks."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def strength(score: float, top: float) -> str:
    """RRF scores sit in a narrow band (roughly 0.013-0.033 for two
    retrievers), so the absolute value tells a reader nothing and the ratio
    to the top hit tells them everything. Buckets, not numbers."""
    ratio = score / top if top else 0.0
    return next(label for label, floor in _STRENGTH if ratio >= floor)


def highlight_quote(text: str, quote: str) -> Markup:
    """Bold the verified quote inside the full provision text.

    Escaping order is the same rule as `citation_chips`, for the same reason:
    this function marks its output safe, so it escapes everything first and
    only then inserts markup. Doing it the other way would escape the <mark>
    into visible angle brackets, or open an injection path through text a
    user can influence via their query.

    Tokens are joined with \\s+ because the source carries XML line breaks
    while a faithful quote reflows them — the exact difference the validator
    forgives. Matching token-wise is how you locate the span in the ORIGINAL
    text rather than in a normalised copy that has no usable offsets.
    """
    safe = str(escape(text))
    tokens = [re.escape(t) for t in str(escape(quote)).split()]

    # Below a few words, bolding is noise rather than signal — "the student"
    # tells the reader nothing. The validator deliberately does NOT reject
    # short quotes, because span length is a usefulness question and not a
    # truthfulness one. This is where that decision lands.
    if len(tokens) < _MIN_QUOTE_TOKENS:
        return Markup(safe)  # noqa: S704 - escaped first, see docstring

    pattern = re.compile(r"\s+".join(tokens))
    # count=1: the same clause can recur in a long provision, and marking
    # every instance implies the answer leaned on all of them.
    return Markup(pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe, count=1))  # noqa: S704


def compare_dates(
    citations: Sequence[Citation],
    *,
    as_of: date,
    corpus_since: date,
    limit: int = 3,
) -> list[date]:
    """Dates where a cited provision reads differently from `as_of`.

    For each cited provision that began after the corpus starts, offer the day
    BEFORE its effective_from — the last day its predecessor was in force.
    Offering effective_from itself is the off-by-one that makes this feature
    look broken: that date is the first day of the version already on screen,
    so the comparison returns the same text twice.
    """
    out: set[date] = set()
    for c in citations:
        began = date.fromisoformat(c.effective_from)
        if began > corpus_since:
            out.add(max(began - timedelta(days=1), corpus_since))
        if c.effective_to:
            out.add(date.fromisoformat(c.effective_to))

    out.discard(as_of)
    # Nearest first: the most recent change is the one most likely to be the
    # reason the answer reads as it does.
    return sorted(out, reverse=True)[:limit]
