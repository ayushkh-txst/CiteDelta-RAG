"""Run an index against a dataset and report percentiles, not means."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

from citedelta.bench.datasets import Dataset
from citedelta.bench.metrics import GroundTruth, recall_by_id, recall_with_ties
from citedelta.index.vector import VectorIndex

log = structlog.get_logger(__name__)

WARMUP_QUERIES = 50
TIMED_REPEATS = 3


@dataclass
class BenchmarkResult:
    dataset: str
    dataset_size: int
    index: str
    effort: int | None
    k: int
    recall: float  # tie-aware — the headline number
    recall_by_id: float  # naive, for the comparison
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    build_seconds: float
    memory_mb: float
    n_queries: int


def measure(
    index: VectorIndex,
    dataset: Dataset,
    truth: GroundTruth,
    *,
    k: int = 10,
    effort: int | None = None,
    build_seconds: float = 0.0,
) -> BenchmarkResult:
    """One (index, effort) point on the curve.

    Methodology, stated because a benchmark without one is an anecdote:
      * WARMUP first. The first queries pay for page faults, lazily-built
        NumPy buffers and a cold cache. Timing them measures the OS.
      * PERCENTILES, not means. One 40 ms stall hides completely in a mean
        over 500 queries and is exactly what a user notices.
      * Single-threaded, one query at a time. Concurrency is a separate
        load test; mixing the two would conflate index cost with pool cost.
      * Repeat the whole query set TIMED_REPEATS times for a stable tail.
    """
    for q in dataset.queries[:WARMUP_QUERIES]:
        index.search(q, k, effort=effort)

    latencies: list[float] = []
    recalls: list[float] = []
    recalls_id: list[float] = []

    for repeat in range(TIMED_REPEATS):
        for i, q in enumerate(dataset.queries):
            t0 = time.perf_counter()
            hits = index.search(q, k, effort=effort)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if repeat == 0:  # recall is deterministic; measure once
                recalls.append(recall_with_ties(hits, truth.distances[i], k))
                recalls_id.append(recall_by_id(hits, truth.ids[i], k))

    latencies.sort()
    total_seconds = sum(latencies) / 1000.0

    return BenchmarkResult(
        dataset=dataset.name,
        dataset_size=dataset.size,
        index=index.name,
        effort=effort,
        k=k,
        recall=statistics.fmean(recalls),
        recall_by_id=statistics.fmean(recalls_id),
        qps=len(latencies) / total_seconds if total_seconds else 0.0,
        p50_ms=latencies[int(0.50 * (len(latencies) - 1))],
        p95_ms=latencies[int(0.95 * (len(latencies) - 1))],
        p99_ms=latencies[int(0.99 * (len(latencies) - 1))],
        build_seconds=build_seconds,
        memory_mb=index.memory_bytes() / 1e6,
        n_queries=len(dataset.queries),
    )


def build_timed(index: VectorIndex, dataset: Dataset) -> float:
    t0 = time.perf_counter()
    index.build(dataset.ids, dataset.vectors)
    elapsed = time.perf_counter() - t0
    log.info("bench.built", index=index.name, dataset=dataset.name, seconds=round(elapsed, 2))
    return elapsed


def save_results(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    log.info("bench.saved", path=str(path), rows=len(results))


def as_markdown(results: list[BenchmarkResult]) -> str:
    head = (
        "| dataset | n | index | effort | recall@10 | recall(id) | QPS "
        "| p50 ms | p95 ms | p99 ms | build s | MB |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {r.dataset} | {r.dataset_size} | {r.index} "
        f"| {r.effort if r.effort is not None else '—'} "
        f"| {r.recall:.3f} | {r.recall_by_id:.3f} | {r.qps:.0f} | {r.p50_ms:.2f} "
        f"| {r.p95_ms:.2f} | {r.p99_ms:.2f} | {r.build_seconds:.1f} | {r.memory_mb:.1f} |\n"
        for r in results
    )
    return head + rows
