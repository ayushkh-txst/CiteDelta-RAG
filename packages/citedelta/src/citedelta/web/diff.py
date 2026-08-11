"""Word-level diff between two answers.

Word-level rather than character-level: character diffs on prose produce
shredded highlighting that is harder to read than no highlighting at all.
Word-level tracks how people actually read a changed sentence.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from markupsafe import Markup, escape

_TOKEN = re.compile(r"\s+|\w+|[^\w\s]")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def diff_pair(old: str, new: str) -> tuple[Markup, Markup]:
    """Return (old_html, new_html) with <del>/<ins> spans.

    Escaping happens per token, before any markup is introduced — same rule as
    the citation-chip filter, and for the same reason. Both halves are marked
    safe on the way out, so nothing unescaped may reach them.
    """
    a, b = _tokens(old), _tokens(new)
    matcher = SequenceMatcher(None, a, b, autojunk=False)

    left: list[str] = []
    right: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_chunk = str(escape("".join(a[i1:i2])))
        new_chunk = str(escape("".join(b[j1:j2])))
        if tag == "equal":
            left.append(old_chunk)
            right.append(new_chunk)
        elif tag == "delete":
            left.append(f"<del>{old_chunk}</del>")
        elif tag == "insert":
            right.append(f"<ins>{new_chunk}</ins>")
        else:  # replace
            left.append(f"<del>{old_chunk}</del>")
            right.append(f"<ins>{new_chunk}</ins>")

    return Markup("".join(left)), Markup("".join(right))  # noqa: S704 - per-token escape, see docstring
