"""The cite-or-refuse prompt.

Two rules govern everything in this file:

1. The model sees ONLY admissible text. Nothing filtered out at `as_of` can be
   cited, because it was never in the context to cite. This is the real
   temporal guarantee; the validator is the check that it held.
2. The model cites ids WE assigned. Not citation paths, not section numbers —
   opaque integers from the retrieved set. A model can plausibly invent
   "8 CFR 214.2(f)(10)" because it has seen a thousand of them. Inventing an
   integer that happens to be in a set of ten we just chose is far less likely,
   and trivially checkable when it happens.
"""

from __future__ import annotations

from typing import Any

from citedelta.answer.models import Citation

SYSTEM_PROMPT = """\
You answer questions about United States immigration regulations, using only \
the excerpts provided in the user message.

Each excerpt is labelled with a numeric id in square brackets, its citation \
path, and the date range during which that text was in force.

Rules:

- Use only the provided excerpts. Do not use anything you know about \
immigration law from elsewhere, even if you are confident it is correct. The \
excerpts are the regulation as it stood on the date asked about; your training \
data is not.
- Cite by id. Every factual claim in your answer must be followed by the id or \
ids of the excerpts supporting it, written as [3] or [3][7].
- Cite only ids that appear in the excerpts below. Never write an id that was \
not given to you.
- If the excerpts do not answer the question, set "sufficient" to false and \
leave "answer" empty. This is a good outcome, not a failure. Do not assemble a \
partial answer from loosely related text.
- Do not give legal advice, recommend a course of action, or estimate an \
individual's eligibility. Describe what the regulation says.
- Be direct. No preamble, no restating the question.\
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": (
                "True only if the provided excerpts contain enough to answer the question directly."
            ),
        },
        "answer": {
            "type": "string",
            "description": (
                "The answer, with inline id citations like [3]. Empty string "
                "when sufficient is false."
            ),
        },
        "citation_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Every excerpt id used, in order of first use.",
        },
    },
    "required": ["sufficient", "answer", "citation_ids"],
    "additionalProperties": False,
}


def build_user_message(query: str, as_of: str, candidates: list[Citation]) -> str:
    """Render the retrieved excerpts plus the question.

    The as-of date is stated explicitly even though the excerpts are already
    filtered to it. Belt and braces: it tells the model the answer is
    time-scoped, which measurably reduces the urge to add "note that this
    changed in 2021" from training data.
    """
    blocks = []
    for c in candidates:
        blocks.append(
            f"[{c.chunk_id}] {c.citation_path}\nin force: {c.in_force_label}\n{c.text.strip()}"
        )
    excerpts = "\n\n---\n\n".join(blocks)
    return (
        f"Question: {query}\n\n"
        f"Answer as the regulation stood on {as_of}.\n\n"
        f"Excerpts:\n\n{excerpts}"
    )
