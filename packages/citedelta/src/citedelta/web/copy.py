"""User-facing refusal copy.

Kept in one file because this is the product's voice in its least forgiving
moment. A refusal is the screen where a user decides whether the system is
careful or broken, and the difference is entirely wording.

Every entry answers two questions: what happened, and what can I do now. A
refusal that only answers the first is a dead end.
"""

from __future__ import annotations

from markupsafe import Markup

REFUSAL_LABELS = {
    "no_admissible_source": "No admissible source",
    "low_confidence": "No close match",
    "insufficient_evidence": "Insufficient evidence",
    "fabricated_citation": "Answer discarded — unverified citation",
    "provider_refused": "Declined",
    "malformed_response": "Could not read the response",
}

REFUSAL_HELP = {
    "no_admissible_source": Markup(
        "Nothing in the indexed corpus was in force on that date and relevant "
        "to the question. Things to try:"
        "<ul><li>Move the as-of date later — the provision may not have "
        "existed yet</li>"
        "<li>Ask about a related provision that did exist then</li></ul>"
    ),
    "low_confidence": Markup(
        "Retrieval found text, but nothing scored well enough to answer from. "
        "Rephrasing with the specific terms used in the regulation "
        "(for example <em>practical training</em> rather than <em>work "
        "permit</em>) usually helps."
    ),
    "insufficient_evidence": Markup(
        "The provisions in force on this date don't cover this question. "
        "CiteDelta indexes <strong>8 CFR Part 214</strong> only — nonimmigrant "
        "classes. Petitions, fees, and adjustment of status are outside the "
        "corpus."
    ),
    "fabricated_citation": Markup(
        "An answer was generated, but one of its citations could not be "
        "verified against the retrieved sources, so the whole answer was "
        "discarded rather than shown."
        "<p>Every citation must exist, must have been among the passages "
        "retrieved for this question, and must have been in force on the "
        "selected date. Showing the remainder would leave claims standing with "
        "nothing behind them.</p>"
        "<p>The retrieval trace below is still accurate — the sources are "
        "real; the summary of them was not trustworthy.</p>"
    ),
    "provider_refused": Markup(
        "The language model declined to answer this question. The retrieved "
        "sources below are still shown."
    ),
    "malformed_response": Markup(
        "The model's response could not be parsed. This is a bug rather than a "
        "judgement about your question — retrying usually works."
    ),
}
