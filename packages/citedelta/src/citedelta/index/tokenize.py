"""Text → terms. Tuned so legal identifiers survive."""

from __future__ import annotations

import re
import unicodedata

# Keep internal dots and hyphens so the tokens that MATTER in this corpus stay
# whole: 214.2, i-20, f-1, h-1b, 8. A naive [a-z]+ tokenizer shatters
# "8 CFR 214.2(f)(10)" into "cfr" plus a pile of meaningless digits, and then
# a user searching the exact citation of their own visa rule gets nothing.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# Deliberately short. BM25 already discounts common terms via IDF — a term in
# 90% of documents scores near zero without being removed. These are the ones
# that appear in nearly every CFR paragraph and only cost postings space.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "shall",
        "such",
        "any",
    }
)


def tokenize(text: str) -> list[str]:
    """Normalize, lowercase, split, drop stopwords.

    NFKC first: the corpus contains typographic quotes, non-breaking spaces and
    full-width digits. Without normalization the same word indexes as two
    different terms depending on which paragraph it came from.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [t for t in _TOKEN.findall(normalized) if t not in _STOPWORDS]
