"""What kind of thing did the user just say?

Deterministic on purpose. A greeting must cost nothing, return instantly,
and be testable without a network call — three properties a model call
cannot give you. The pattern set is small and curated rather than clever;
the goal is to catch the obvious cases with certainty, not to classify
everything.

Anything not obviously conversational falls through to retrieval, where the
model makes the harder judgement — no retrieval score can separate 'out of
scope' from 'not relevant', so that call belongs to the model, not to this
classifier.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Intent(StrEnum):
    GREETING = "greeting"
    PASSTHROUGH = "passthrough"


# Anchored, whole-utterance patterns. A question that merely CONTAINS
# "thanks" is still a question — "thanks, now what about STEM OPT?" must
# not short-circuit to a pleasantry.
_GREETING = re.compile(
    r"""^\s*(?:
        (?:hi|hey|hello|yo|howdy|greetings)
      | (?:hi|hey)\s+there
      | good\s+(?:morning|afternoon|evening|day)
      | how(?:'?s|\s+is|\s+are)\s+(?:it\s+going|you|things|your\s+day)
      | how\s+do\s+you\s+do
      | what'?s\s+up
      | (?:thanks|thank\s+you|cheers|ta)
      | (?:bye|goodbye|see\s+you|later)
      | who\s+are\s+you
      | what\s+(?:can|do)\s+you\s+do
      | help
    )
    [\s!.,?—-]*$""",
    re.IGNORECASE | re.VERBOSE,
)


def classify(text: str) -> Intent:
    """Greetings and small talk, or everything else."""
    if _GREETING.match(text.strip()):
        return Intent.GREETING
    return Intent.PASSTHROUGH
