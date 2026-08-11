from __future__ import annotations

from citedelta.web.diff import diff_pair


def test_equal_text_has_no_markup() -> None:
    left, right = diff_pair("The quick brown fox.", "The quick brown fox.")
    assert "<del>" not in str(left)
    assert "<ins>" not in str(right)
    assert str(left) == str(right)


def test_changed_text_marks_del_and_ins() -> None:
    left, right = diff_pair("a b c", "a x c")
    assert "<del>b</del>" in str(left)
    assert "<ins>x</ins>" in str(right)


def test_html_is_escaped_before_marking() -> None:
    left, _ = diff_pair("look <script>alert(1)</script>", "look safe")
    assert "<script>" not in str(left)
    assert "&lt;script&gt;" in str(left)


def test_common_words_are_not_junk() -> None:
    """autojunk=False must be honoured: 'the' is common in prose, and a change
    to a common word must still be marked rather than silently dropped."""
    old = "I like the the the the"
    new = "I like a the the the"
    left, right = diff_pair(old, new)
    assert "<del>" in str(left) and "the" in str(left)
    assert "<ins>" in str(right) and "a" in str(right)
