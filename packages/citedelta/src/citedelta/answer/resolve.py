"""Turn a follow-up into a standalone question, and pull out any date change.

Runs only when a conversation has prior turns. Uses claude-haiku-4-5 through
the same `Completions` port as the answer model — the first time a second
model at a different price tier goes through the dated rate table, and the
proof that the port is real rather than decorative.

The model's output is NOT trusted. `_safe_as_of` clamps any date it returns
into the corpus window and discards anything unparseable, exactly the same
discipline the citation validator applies to citations: the model proposes,
deterministic code decides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import structlog

from citedelta.api.traces import PriorTurn
from substrate.llm import CompletionRequest, Completions, Message, Role

log = structlog.get_logger(__name__)

RESOLVER_MODEL = "claude-haiku-4-5"
"""Deliberately not the answer model. This task is short, mechanical, and
high-volume — rewriting one sentence given three lines of history. Haiku is
5x cheaper on input and 5x on output, and a resolve call runs around
$0.0008 against roughly $0.027 for the answer it precedes. Using the answer
model here would double the cost of a follow-up to buy nothing."""

SYSTEM = """\
You rewrite follow-up questions about US immigration regulations so they can \
stand alone, and you detect when the user wants a different point in time.

You are given the recent conversation and the newest message. Return:

- is_followup: true only if the newest message depends on the conversation to \
make sense. A complete question on a new topic is not a follow-up, even in the \
middle of a conversation.
- standalone_question: the newest message rewritten to stand on its own, \
carrying forward whatever subject the user is referring to. If it already \
stands alone, return it unchanged.
- as_of: an ISO date (YYYY-MM-DD) if the user is asking about a different point \
in time, otherwise an empty string.

Rules for as_of:
- "in 2019", "back in 2019" -> pick a date inside that year.
- "then", "at that time", "back then" -> the date of the turn being referred to.
- "now", "today", "currently", "these days" -> today's date, given below.
- "before the 2020 change", "after that amendment" -> your best single date.
- If the user says nothing about time, return an empty string. Do not guess.

Never answer the question. Only rewrite it.\
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_followup": {"type": "boolean"},
        "standalone_question": {"type": "string"},
        # A plain string with "" for absent, rather than a nullable type.
        # Structured outputs support anyOf, but an empty-string sentinel has
        # one less way to go wrong and reads identically at the call site.
        "as_of": {
            "type": "string",
            "description": "ISO YYYY-MM-DD, or empty string if unchanged.",
        },
    },
    "required": ["is_followup", "standalone_question", "as_of"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Resolution:
    standalone_question: str
    as_of: date | None
    is_followup: bool


def _safe_as_of(raw: str, *, corpus_since: date, today: date) -> date | None:
    """Parse and clamp. Never trust a date the model invented.

    Three failure modes this closes:
      - unparseable text ("sometime in 2019") -> None, keep the current date
      - a date before the corpus starts -> clamped to corpus_since, so the
        user gets the earliest law we actually hold rather than a refusal
        that looks like a bug
      - a date in the future -> clamped to today
    """
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        log.warning("resolve.unparseable_date", raw=text[:40])
        return None
    return min(max(parsed, corpus_since), today)


def _history_block(history: list[PriorTurn]) -> str:
    lines = []
    for turn in history:
        lines.append(f"[{turn.as_of.isoformat()}] user: {turn.query}")
        if turn.resolved_query != turn.query:
            lines.append(f"           (searched: {turn.resolved_query})")
        lines.append(
            "           assistant: answered" if turn.answered else "           assistant: declined"
        )
    return "\n".join(lines)


async def resolve_followup(
    llm: Completions,
    *,
    question: str,
    history: list[PriorTurn],
    current_as_of: date,
    corpus_since: date,
    today: date,
    run_id: str = "adhoc",
) -> Resolution:
    """No history means nothing to resolve — return the question untouched
    without spending a call. The first turn of every conversation takes this
    path, which is most turns."""
    if not history:
        return Resolution(question, None, is_followup=False)

    user = (
        f"Today is {today.isoformat()}. "
        f"The conversation is currently set to {current_as_of.isoformat()}.\n"
        f"Regulation history is available from {corpus_since.isoformat()} onward.\n\n"
        f"Conversation so far:\n{_history_block(history)}\n\n"
        f"Newest message: {question}"
    )

    response = await llm.complete(
        CompletionRequest(
            model=RESOLVER_MODEL,
            system=SYSTEM,
            messages=(Message(Role.USER, user),),
            max_tokens=400,
            json_schema=SCHEMA,
            run_id=run_id,
        )
    )

    # A refused or unreadable resolution must not take the turn down with
    # it. Falling back to the raw question means retrieval may do poorly on
    # a follow-up — degraded, not broken.
    if response.refused:
        log.warning("resolve.provider_refused")
        return Resolution(question, None, is_followup=False)

    try:
        payload = json.loads(response.text)
        standalone = str(payload["standalone_question"]).strip() or question
        resolved_as_of = _safe_as_of(str(payload["as_of"]), corpus_since=corpus_since, today=today)
        is_followup = bool(payload["is_followup"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        log.warning("resolve.malformed", error=str(exc))
        return Resolution(question, None, is_followup=False)

    log.info(
        "resolve.ok",
        is_followup=is_followup,
        as_of=resolved_as_of.isoformat() if resolved_as_of else None,
        cost_usd=str(response.cost_usd),
    )
    return Resolution(standalone, resolved_as_of, is_followup)
