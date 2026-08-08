"""Parser behaviour, pinned against the shapes real CFR text takes."""

from __future__ import annotations

from citedelta.ecfr.parse import _designators, _PathTracker, parse_part


def _paths(paragraph_texts: list[str]) -> list[tuple[str, ...]]:
    t = _PathTracker()
    out = []
    for text in paragraph_texts:
        for tok in _designators(text):
            t.advance(tok)
        out.append(t.path())
    return out


def test_em_dash_opens_a_nested_designator() -> None:
    assert _designators("(ii) Optional practical training—(A) General. Consistent") == [
        "ii",
        "A",
    ]


def test_plain_leading_designator() -> None:
    assert _designators("(10) Practical training. Practical training may be") == ["10"]


def test_no_designator() -> None:
    assert _designators("Table 2 to Paragraph (f)—Paragraph Contents") == []


def test_sibling_closes_deeper_levels() -> None:
    """(B) after (A)(1)(2)(3) is a sibling of (A), not a child of (3)."""
    paths = _paths(
        [
            "(f) Students in colleges",
            "(10) Practical training.",
            "(ii) Optional practical training—(A) General.",
            "(1) During the student's annual vacation",
            "(2) While school is in session",
            "(3) After completion of the course of study",
            "(B) Termination of practical training.",
        ]
    )
    assert paths[-2] == ("f", "10", "ii", "A", "3")
    assert paths[-1] == ("f", "10", "ii", "B")


def test_roman_one_does_not_eat_the_top_level() -> None:
    """(i) is both 'the letter after (h)' and 'roman numeral one'. Treating it
    as a letter-successor made (h)'s entire subtree reparent to (i), producing
    citations like 8 CFR 214.2(ii)(E)(2)... with the (h) gone."""
    paths = _paths(
        [
            "(h) Temporary employees.",
            "(1) Classification.",
            "(i) Types of H classification.",
            "(A) An H-1B classification applies to",
        ]
    )
    assert paths[-1] == ("h", "1", "i", "A")
    assert all(p[0] == "h" for p in paths)


def test_parse_part_produces_chunks_with_citations() -> None:
    xml = b"""<?xml version="1.0"?>
    <DIV5 N="214" TYPE="PART">
      <DIV8 N="214.2" TYPE="SECTION">
        <HEAD>&#167; 214.2 Special requirements.</HEAD>
        <P>(f) Students in colleges and universities and other institutions.</P>
        <P>(10) Practical training may be authorized to an F-1 student.</P>
      </DIV8>
    </DIV5>"""
    sections = parse_part(xml, citation_prefix="8 CFR")
    assert len(sections) == 1
    assert sections[0].section == "214.2"
    assert "Special requirements" in sections[0].heading
    joined = " ".join(c.text for c in sections[0].chunks)
    assert "Practical training" in joined
    assert sections[0].chunks[0].citation_path.startswith("8 CFR 214.2(f)")
