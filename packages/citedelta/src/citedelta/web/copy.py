"""User-facing copy for every non-answer outcome.

One file because this is the product's voice at its least forgiving moment.
A refusal is where someone decides whether the system is careful or broken,
and the difference is entirely wording.

Every entry answers two questions: what happened, and what can I do now. One
that answers only the first is a dead end.
"""

from __future__ import annotations

from markupsafe import Markup

GREETING_REPLY = (
    "Doing well — ready when you are. I answer questions about **8 CFR Part "
    "214**, the rules for nonimmigrant classes: F-1 student status, practical "
    "training, transfers, grace periods. I can answer as the regulation stood "
    "on any date since December 2016, so if you filed something in 2019, ask "
    "me about 2019."
)

REFUSAL_LABELS = {
    "greeting": "",  # never rendered — greetings take their own branch
    "out_of_scope": "Outside this regulation",
    "no_admissible_source": "Nothing in force on that date",
    "low_confidence": "No close match",
    "insufficient_evidence": "Not covered on that date",
    "fabricated_citation": "Answer discarded — quote not verified",
    "provider_refused": "Declined",
    "malformed_response": "Could not read the response",
}

REFUSAL_HELP = {
    "out_of_scope": Markup(
        "I only read <strong>8 CFR Part 214</strong> — nonimmigrant classes — and "
        "I won't answer from general knowledge, because then you couldn't check "
        "me against a source."
        "<ul><li>F-1 status, practical training, STEM extensions</li>"
        "<li>School transfers, reinstatement, grace periods</li>"
        "<li>Any of those, as they stood on a date you pick</li></ul>"
    ),
    "no_admissible_source": Markup(
        "Nothing in the corpus was in force on that date and relevant to the "
        "question — often because the provision didn't exist yet."
        "<ul><li>Move the date later</li>"
        "<li>Ask about a related provision that did exist then</li></ul>"
    ),
    "low_confidence": Markup(
        "Retrieval found text, but nothing scored well enough to answer from. "
        "Using the regulation's own wording usually helps — <em>practical "
        "training</em> rather than <em>work permit</em>."
    ),
    "insufficient_evidence": Markup(
        "This is in scope, but the provisions in force on that date don't "
        "address it. Try a different date, or a narrower question."
    ),
    "fabricated_citation": Markup(
        "Every citation has to exist, have been among the passages retrieved for "
        "your question, have been in force on your date, and quote the source "
        "word for word. One failed, so the whole answer went — showing the rest "
        "would leave claims standing with nothing behind them."
        "<p>Asking again usually works; this is a generation failure, not a "
        "judgement about your question.</p>"
    ),
    "provider_refused": Markup(
        "The language model declined to answer. The sources retrieved for your "
        "question were still valid — try rephrasing."
    ),
    "malformed_response": Markup(
        "The model's reply couldn't be parsed. That's a bug on my side rather "
        "than anything about your question — asking again usually works."
    ),
}
