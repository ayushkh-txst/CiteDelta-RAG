"""eCFR XML → sections → citation-pathed chunks."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import structlog

from citedelta.ecfr.models import ParsedChunk, ParsedSection

log = structlog.get_logger(__name__)

_LEADING = re.compile(r"^\(([A-Za-z0-9]+)\)\s*")
_EMDASH = re.compile(r"^[^—]{0,120}—\s*\(([A-Za-z0-9]+)\)\s*")

_ROMAN_CHARS = set("ivxlcdm")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

# a new depth is only opened by the first element of its sequence
_FIRST_OF_SEQUENCE = frozenset({"1", "a", "A", "i", "I"})

# CFR nesting bottoms out around 6 levels; clamp anything past 8 as a
# parse artifact rather than letting the stack run away
_MAX_DEPTH = 8

# chunk sizing in characters: small paragraphs get merged so a chunk carries
# enough context to be worth embedding later
_TARGET_CHARS = 900
_MAX_CHARS = 1800


def _roman_to_int(tok: str) -> int | None:
    low = tok.lower()
    if not low or any(c not in _ROMAN_CHARS for c in low):
        return None
    total = prev = 0
    for ch in reversed(low):
        v = _ROMAN_VALUES[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _same_family(a: str, b: str) -> bool:
    if a.isdigit() != b.isdigit():
        return False
    return True if a.isdigit() else a.isupper() == b.isupper()


def _is_successor(prev: str, cur: str) -> bool:
    if prev.isdigit() and cur.isdigit():
        return int(cur) == int(prev) + 1
    if (
        len(prev) == len(cur) == 1
        and prev.isalpha()
        and cur.isalpha()
        and prev.isupper() == cur.isupper()
        and ord(cur) == ord(prev) + 1
    ):
        return True
    p, c = _roman_to_int(prev), _roman_to_int(cur)
    if p is not None and c is not None and prev.isupper() == cur.isupper():
        return c == p + 1
    return False


def _designators(text: str) -> list[str]:
    """Every designator this paragraph introduces, in order."""
    toks: list[str] = []
    rest = text
    while (m := _LEADING.match(rest)) and len(toks) < _MAX_DEPTH:
        toks.append(m.group(1))
        rest = rest[m.end() :]
    # "(ii) Optional practical training—(A) General. ..." opens (A) too.
    while (m := _EMDASH.match(rest)) and len(toks) < _MAX_DEPTH:
        toks.append(m.group(1))
        rest = rest[m.end() :]
    return toks


class _PathTracker:
    """The designator stack, rebuilt paragraph by paragraph."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.expected_top = "a"
        self.clamped = 0

    def advance(self, tok: str) -> None:
        # 1. Top level is anchored to (a),(b),(c),... in order. Nothing else
        #    may reset the stack — this stops a roman (i) deep inside (h)'s
        #    subtree from being mistaken for the top-level (i).
        if tok == self.expected_top:
            self.stack = [tok]
            self.expected_top = chr(ord(tok) + 1)
            return

        # 2. A first-of-sequence token opens a new depth.
        if tok in _FIRST_OF_SEQUENCE:
            self._push(tok)
            return

        # 3. Otherwise it continues an existing level. Search deepest-first so
        #    (B) after (A)(1)(2)(3) closes the digits and lands beside (A) —
        #    never at level 0, which is why range() stops at 1.
        for lvl in range(len(self.stack) - 1, 0, -1):
            if _same_family(self.stack[lvl], tok) and _is_successor(self.stack[lvl], tok):
                del self.stack[lvl + 1 :]
                self.stack[lvl] = tok
                return

        self._push(tok)

    def _push(self, tok: str) -> None:
        if len(self.stack) >= _MAX_DEPTH:
            self.clamped += 1
            return
        self.stack.append(tok)

    def path(self) -> tuple[str, ...]:
        return tuple(self.stack)


def _clean(el: ET.Element) -> str:
    """All descendant text, whitespace collapsed. <E>, <I> etc. are inline."""
    return " ".join("".join(el.itertext()).split())


def parse_part(xml_bytes: bytes, *, citation_prefix: str) -> list[ParsedSection]:
    """Parse one point-in-time snapshot of a CFR part."""
    root = ET.fromstring(xml_bytes)  # noqa: S314 - trusted government source
    sections: list[ParsedSection] = []
    total_clamped = 0

    for div in root.iter("DIV8"):
        if div.attrib.get("TYPE") != "SECTION":
            continue
        number = div.attrib.get("N", "").strip()
        if not number:
            continue

        head_el = div.find("HEAD")
        heading = _clean(head_el) if head_el is not None else ""

        tracker = _PathTracker()
        paragraphs: list[tuple[str, str]] = []  # (citation_path, text)

        for p in div.iter("P"):
            text = _clean(p)
            if not text:
                continue
            for tok in _designators(text):
                tracker.advance(tok)
            suffix = "".join(f"({t})" for t in tracker.path())
            paragraphs.append((f"{citation_prefix} {number}{suffix}", text))

        total_clamped += tracker.clamped
        sections.append(
            ParsedSection(
                section=number,
                heading=heading,
                chunks=_chunk(paragraphs),
            )
        )

    if total_clamped:
        log.warning("parse.depth_clamped", paragraphs=total_clamped)
    return sections


def _chunk(paragraphs: list[tuple[str, str]]) -> list[ParsedChunk]:
    """Merge consecutive paragraphs into retrieval-sized units.

    The chunk keeps the citation path of its FIRST paragraph, the narrowest
    one that covers all of it.
    """
    chunks: list[ParsedChunk] = []
    buf: list[str] = []
    buf_path = ""
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_path, buf_len
        if buf:
            chunks.append(
                ParsedChunk(ordinal=len(chunks), citation_path=buf_path, text=" ".join(buf))
            )
            buf, buf_path, buf_len = [], "", 0

    for path, text in paragraphs:
        if buf and (buf_len >= _TARGET_CHARS or buf_len + len(text) > _MAX_CHARS):
            flush()
        if not buf:
            buf_path = path
        buf.append(text)
        buf_len += len(text) + 1

    flush()
    return chunks
