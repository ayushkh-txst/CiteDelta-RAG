from __future__ import annotations

from datetime import date

from citedelta.answer.models import Citation
from citedelta.web.filters import compare_dates, highlight_quote, strength

CHUNK = (
    "(iv) Preparation for departure. An F-1 student who has completed a\n"
    "course of study will be allowed an additional 60-day period."
)


def _citation(*, effective_from: str, effective_to: str | None) -> Citation:
    return Citation(
        chunk_id=1,
        citation_path="8 CFR 214.2(f)(5)(iv)",
        effective_from=effective_from,
        effective_to=effective_to,
        text="x",
        rrf_score=0.03,
    )


def test_quote_is_bolded_across_a_line_break() -> None:
    """The span must be found in the ORIGINAL text, which has XML newlines."""
    out = str(highlight_quote(CHUNK, "completed a course of study"))
    assert "<mark>completed a\ncourse of study</mark>" in out


def test_absent_quote_leaves_the_text_alone() -> None:
    out = str(highlight_quote(CHUNK, "words that are not there"))
    assert "<mark>" not in out


def test_short_quote_is_not_highlighted() -> None:
    """Bolding two words is noise. Short quotes are allowed, but not worth
    highlighting."""
    assert "<mark>" not in str(highlight_quote(CHUNK, "the student"))


def test_html_in_the_source_is_escaped_not_rendered() -> None:
    out = str(highlight_quote("Text with <script>alert(1)</script> inside", "nope"))
    assert "&lt;script&gt;" in out
    assert "<script>" not in out


def test_only_the_first_occurrence_is_marked() -> None:
    text = "the rule applied on this date and the rule applied on that date"
    assert str(highlight_quote(text, "the rule applied on")).count("<mark>") == 1


def test_strength_buckets_scale_to_the_top_hit() -> None:
    assert strength(0.0328, 0.0328) == "strongest"
    assert strength(0.0160, 0.0328) == "moderate"
    assert strength(0.0135, 0.0328) == "weak"


def test_compare_dates_offer_the_day_before_a_change() -> None:
    """The off-by-one: effective_from itself is the first day of the version
    already on screen, so it would compare the text against itself."""
    c = _citation(effective_from="2026-07-17", effective_to=None)
    dates = compare_dates([c], as_of=date(2026, 8, 13), corpus_since=date(2016, 12, 23))
    assert date(2026, 7, 16) in dates
    assert date(2026, 7, 17) not in dates


def test_compare_dates_exclude_the_current_date() -> None:
    c = _citation(effective_from="2026-07-17", effective_to=None)
    assert date(2026, 7, 16) not in compare_dates(
        [c], as_of=date(2026, 7, 16), corpus_since=date(2016, 12, 23)
    )


def test_provisions_from_the_corpus_start_offer_nothing() -> None:
    """Nothing precedes the corpus, so there is no predecessor to compare."""
    c = _citation(effective_from="2016-12-23", effective_to=None)
    assert compare_dates([c], as_of=date(2026, 8, 13), corpus_since=date(2016, 12, 23)) == []
