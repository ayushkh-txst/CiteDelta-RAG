"""Run the suite, write a scorecard."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

import structlog

from citedelta.eval.cases import CASES
from citedelta.eval.graders import CaseScore, grade

log = structlog.get_logger(__name__)


async def run_eval(state: Any) -> dict[str, Any]:
    from citedelta.api.app import AskRequest, _run_ask

    scores: list[CaseScore] = []
    for case in CASES:
        payload = await _run_ask(state, AskRequest(query=case.query, as_of=case.as_of))
        score = grade(case, payload["result"], payload["candidates"])
        scores.append(score)
        log.info("eval.case", id=case.id, passed=score.passed)

    by_class: dict[str, list[CaseScore]] = defaultdict(list)
    for s in scores:
        by_class[s.cls].append(s)

    def rate(items: list[CaseScore], attr: str) -> float:
        return sum(getattr(i, attr) for i in items) / len(items) if items else 0.0

    return {
        "n": len(scores),
        "verified": sum(1 for c in CASES if c.verified_by),
        "overall": {
            "recall_at_5": rate(scores, "retrieved"),
            "citation_validity": rate(scores, "citations_valid"),
            "refusal_accuracy": rate(scores, "refusal_correct"),
            "temporal_accuracy": rate(scores, "temporal_correct"),
            "pass_rate": rate(scores, "passed"),
        },
        "by_class": {
            cls: {
                "n": len(items),
                "recall_at_5": rate(items, "retrieved"),
                "pass_rate": rate(items, "passed"),
            }
            for cls, items in sorted(by_class.items())
        },
        "failures": [asdict(s) for s in scores if not s.passed],
        "p50_latency_ms": sorted(s.latency_ms for s in scores)[len(scores) // 2],
    }
