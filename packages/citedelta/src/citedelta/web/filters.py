"""Jinja filters. The escaping here is load-bearing — read the docstring."""

from __future__ import annotations

import re

from markupsafe import Markup, escape

from citedelta.answer.models import Citation

_CITE = re.compile(r"\[(\d+)\]")


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
