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
- Quote, don't paraphrase. For every id you cite, copy the exact words from \
that excerpt that support your claim — character for character, at least a \
full clause. Do not shorten with ellipses, do not tidy the wording, do not \
join text from two places. Your quote is checked against the source and the \
whole answer is discarded if it does not match.
- If the question is not about immigration regulation at all — general \
knowledge, personal advice, another area of law — set out_of_scope to true and \
leave the answer empty. Do not try to find something adjacent in the excerpts.
- Do not give legal advice, recommend a course of action, or estimate an \
individual's eligibility. Describe what the regulation says.
- Be direct. No preamble, no restating the question.\
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "out_of_scope": {
            "type": "boolean",
            "description": (
                "True if the question is not about US immigration regulation at "
                "all — general knowledge, personal advice, or another area of "
                "law. Distinct from a regulation question these excerpts "
                "happen not to cover."
            ),
        },
        "sufficient": {
            "type": "boolean",
            "description": (
                "True only if the excerpts contain enough to answer directly. "
                "False when the question is in scope but the provisions in "
                "force on this date do not address it."
            ),
        },
        "answer": {
            "type": "string",
            "description": (
                "The answer, with inline id citations like [3]. Empty when "
                "out_of_scope or not sufficient."
            ),
        },
        "citations": {
            "type": "array",
            "description": "One entry per excerpt used, in order of first use.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The excerpt id, exactly as given.",
                    },
                    "quote": {
                        "type": "string",
                        "description": (
                            "The exact words from that excerpt supporting the "
                            "claim, copied character for character. At least a "
                            "full clause. Never paraphrased, never elided with "
                            "ellipses."
                        ),
                    },
                },
                "required": ["id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["out_of_scope", "sufficient", "answer", "citations"],
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
