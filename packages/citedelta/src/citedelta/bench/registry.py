"""Which indexes get benchmarked, and over which effort sweeps."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import structlog

from citedelta.bench.datasets import Dataset, cfr_dedup, cfr_full, random_hard
from citedelta.bench.metrics import compute_ground_truth
from citedelta.bench.runner import BenchmarkResult, build_timed, measure
from citedelta.config import get_settings
from citedelta.embed.corpus import load_corpus_vectors
from citedelta.index.brute import BruteForceIndex
from citedelta.index.hnsw import HNSWIndex
from citedelta.index.ivf import IVFFlatIndex
from citedelta.index.pgvector import PgVectorIndex
from citedelta.index.vector import VectorIndex

log = structlog.get_logger(__name__)

# (factory, effort sweep). None = the index has no accuracy knob.
INDEXES: list[tuple[str, Callable[[], VectorIndex], Sequence[int | None]]] = [
    ("brute-force", BruteForceIndex, [None]),
    ("ivf-flat", IVFFlatIndex, [1, 2, 4, 8, 16, 32, 64, 128]),
    ("hnsw", HNSWIndex, [10, 16, 32, 64, 128, 256]),
    (
        "pgvector-hnsw",
        lambda: PgVectorIndex(get_settings().database_url, probe="hnsw"),
        [10, 16, 32, 64, 128, 256],
    ),
    (
        "pgvector-ivf",
        lambda: PgVectorIndex(get_settings().database_url, probe="ivf"),
        [1, 2, 4, 8, 16, 32, 64, 128],
    ),
]


async def load_datasets(which: str) -> list[Dataset]:
    if which == "random-hard":
        return [random_hard()]

    ids, vectors = await load_corpus_vectors()
    table = {"cfr-full": cfr_full(ids, vectors), "cfr-dedup": cfr_dedup(ids, vectors)}
    if which == "all":
        return [table["cfr-full"], table["cfr-dedup"], random_hard()]
    if which in table:
        return [table[which]]
    msg = f"unknown dataset {which!r}"
    raise ValueError(msg)


async def run_suite(which: str, *, k: int = 10) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for dataset in await load_datasets(which):
        log.info(
            "bench.dataset", name=dataset.name, size=dataset.size, queries=len(dataset.queries)
        )
        # Ground truth once per dataset — a property of the DATA, not of any
        # index. Recomputed per index would risk each being scored against a
        # subtly different truth.
        truth = compute_ground_truth(dataset.vectors, dataset.ids, dataset.queries, k)

        for label, factory, sweep in INDEXES:
            index = factory()
            build_seconds = build_timed(index, dataset)
            for effort in sweep:
                r = measure(index, dataset, truth, k=k, effort=effort, build_seconds=build_seconds)
                log.info(
                    "bench.point",
                    index=label,
                    effort=effort,
                    recall=round(r.recall, 3),
                    qps=round(r.qps),
                )
                results.append(r)

    return results
