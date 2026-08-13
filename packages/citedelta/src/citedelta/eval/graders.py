"""Deterministic graders. No LLM judges — see the eval README for why."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from citedelta.answer.models import AnswerResult, Citation
from citedelta.answer.validator import normalize
from citedelta.eval.cases import EvalCase


@dataclass(frozen=True, slots=True)
class CaseScore:
    case_id: str
    cls: str
    retrieved: bool
    citations_valid: bool
    refusal_correct: bool
    reason_correct: bool
    temporal_correct: bool
    leaked_forbidden: bool
    latency_ms: float

    @property
    def passed(self) -> bool:
        return (
            self.citations_valid
            and self.refusal_correct
            and self.temporal_correct
            and not self.leaked_forbidden
        )


def _matches(path: str, expected: str) -> bool:
    """Match on citation path regardless of which side is more specific.

    The chunker does not always split where a paragraph boundary falls: the
    expected '214.2(f)(15)(i)' may live inside a chunk tagged
    '214.2(f)(15)', or a coarse expectation may match a finer chunk path.
    Match when either normalized path is a prefix of the other, so the eval
    isn't coupled to chunk boundaries in either direction.
    """
    p = path.replace("8 CFR ", "")
    e = expected.replace("8 CFR ", "")
    return p.startswith(e) or e.startswith(p)


def _citations_hold(case: EvalCase, result: AnswerResult, candidates: list[Citation]) -> bool:
    """Re-verify the guarantee from the outside.

    Deliberately a separate implementation from answer/validator.py. The
    eval's job is to check the promise, not to ask the promise-keeper whether
    it kept the promise — sharing the code would let one bug pass both.
    """
    if result.refused:
        return True  # nothing was asserted, so nothing can be unsupported

    shown = {c.chunk_id: c for c in candidates}
    for cited in result.citations:
        source = shown.get(cited.chunk_id)
        if source is None:
            return False
        if not cited.quote.strip():
            return False
        if normalize(cited.quote) not in normalize(source.text):
            return False
    return True


def grade(case: EvalCase, result: AnswerResult, candidates: list[Citation]) -> CaseScore:
    top5 = candidates[:5]

    retrieved = not case.expected_citations or any(
        _matches(c.citation_path, exp) for exp in case.expected_citations for c in top5
    )

    cited = () if result.refused else result.citations

    leaked = any(_matches(c.citation_path, bad) for bad in case.must_not_cite for c in cited)

    # Every cited provision must have been in force at as_of. Independent of
    # the production validator on purpose: this is the eval checking the
    # guarantee, not the system checking itself.
    temporal_ok = all(
        date.fromisoformat(c.effective_from) <= case.as_of
        and (c.effective_to is None or case.as_of < date.fromisoformat(c.effective_to))
        for c in cited
    )

    return CaseScore(
        case_id=case.id,
        cls=case.cls.value,
        retrieved=retrieved,
        citations_valid=_citations_hold(case, result, candidates),
        refusal_correct=result.refused == case.expects_refusal,
        reason_correct=(
            case.expected_reason is None
            or (result.refused and result.reason.value == case.expected_reason)
        ),
        temporal_correct=temporal_ok,
        leaked_forbidden=leaked,
        latency_ms=result.latency_ms,
    )
