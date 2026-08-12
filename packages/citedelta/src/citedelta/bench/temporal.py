"""The measurement CiteDelta exists to make."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np
import structlog

from citedelta.bench.strategies import (
    IN_INDEX,
    POST_FILTER,
    POST_FILTER_OVERFETCH,
    post_filter_search,
    required_overfetch,
)
from citedelta.config import get_settings
from citedelta.embed.corpus import load_corpus_vectors
from citedelta.index.brute import BruteForceIndex
from citedelta.index.lexical import LexicalIndex
from citedelta.index.vector import Ids, VectorIndex, Vectors
from citedelta.ingest import EXTERNAL_ID
from citedelta.store.corpus import CorpusStore
from citedelta.temporal import AdmissibleSet, AsOf
from substrate.db import Database

log = structlog.get_logger(__name__)

# Multipliers on k. At k=10 this sweeps 10 -> 5,000 candidates fetched.
OVERFETCH_LEVELS = (1, 2, 5, 10, 25, 50, 100, 250, 500)

# The admissible id set depends ONLY on the date. Dates repeat heavily in
# real traffic, and rebuilding the set on every request means a Postgres
# round-trip per request that a dict lookup could have served.
_ADMISSIBLE_IDS_CACHE: dict[date, frozenset[int]] = {}


async def _fetch_admissible_ids(as_of: date, conn: Any) -> set[int]:
    document_id = await conn.fetchval(
        "SELECT id FROM documents WHERE external_id = $1", EXTERNAL_ID
    )
    point = AsOf(valid_on=as_of)
    return await CorpusStore(conn).admissible_chunk_ids(int(document_id), point)


async def load_admissible(
    as_of: date, corpus_size: int, *, db: Database | None = None
) -> AdmissibleSet:
    ids = _ADMISSIBLE_IDS_CACHE.get(as_of)
    if ids is None:
        if db is not None:
            async with db.acquire() as conn:
                ids = frozenset(await _fetch_admissible_ids(as_of, conn))
        else:
            async with Database.open(get_settings().database_url) as pool, pool.acquire() as conn:
                ids = frozenset(await _fetch_admissible_ids(as_of, conn))
        _ADMISSIBLE_IDS_CACHE[as_of] = ids
    return AdmissibleSet.from_as_of(set(ids), AsOf(valid_on=as_of), corpus_size)


@dataclass
class CollapseResult:
    as_of: str
    corpus_size: int
    admissible: int
    selectivity: float
    k: int
    n_queries: int
    mean_survivors: float
    zero_result_rate: float
    # overfetch multiplier -> recall@k against the filtered oracle
    recall_by_overfetch: dict[int, float] = field(default_factory=dict)

    @property
    def naive_recall(self) -> float:
        return self.recall_by_overfetch.get(1, 0.0)


def held_out_split(
    ids: Ids, vectors: Vectors, *, n_queries: int = 300, seed: int = 0
) -> tuple[Ids, Vectors, Vectors]:
    """Queries held OUT of the index, so nothing self-matches at distance 0.

    Same discipline as the brute-force benchmark. A query that is itself a
    corpus row would be returned first at distance 0, and whether it survived
    the filter would dominate the result — measuring the split, not the
    strategy.
    """
    rng = np.random.default_rng(seed)
    held = rng.choice(len(ids), n_queries, replace=False)
    keep = np.ones(len(ids), dtype=bool)
    keep[held] = False
    return ids[keep], vectors[keep], vectors[held]


async def measure_collapse(
    as_of: date, *, k: int = 10, n_queries: int = 300, seed: int = 0
) -> CollapseResult:
    """Post-filtering vs. the filtered oracle, on the real corpus."""
    all_ids, all_vectors = await load_corpus_vectors()
    corpus_ids, corpus_vectors, queries = held_out_split(
        all_ids, all_vectors, n_queries=n_queries, seed=seed
    )

    index: VectorIndex = BruteForceIndex()
    index.build(corpus_ids, corpus_vectors)

    admissible = await load_admissible(as_of, index.size)
    mask = index.compile_filter(admissible.ids)

    log.info(
        "collapse.setup",
        as_of=str(as_of),
        corpus=index.size,
        admissible=int(mask.sum()),
        selectivity=round(float(mask.mean()), 4),
    )

    survivors: list[int] = []
    hits: dict[int, int] = dict.fromkeys(OVERFETCH_LEVELS, 0)

    for query in queries:
        # THE ORACLE — exact k nearest admissible neighbours.
        truth = {n.id for n in index.search(query, k, admissible=mask)}

        for overfetch in OVERFETCH_LEVELS:
            got = post_filter_search(index, query, k, admissible, overfetch=overfetch)
            if overfetch == 1:
                survivors.append(len(got))
            hits[overfetch] += len({n.id for n in got} & truth)

    result = CollapseResult(
        as_of=str(as_of),
        corpus_size=index.size,
        admissible=int(mask.sum()),
        selectivity=float(mask.mean()),
        k=k,
        n_queries=len(queries),
        mean_survivors=float(np.mean(survivors)),
        zero_result_rate=float(np.mean([s == 0 for s in survivors])),
        recall_by_overfetch={o: hits[o] / (len(queries) * k) for o in OVERFETCH_LEVELS},
    )

    log.info(
        "collapse.measured",
        naive_recall=round(result.naive_recall, 3),
        zero_result_rate=round(result.zero_result_rate, 3),
    )
    return result


def as_markdown(result: CollapseResult) -> str:
    lines = [
        f"### Post-filter collapse at `as_of = {result.as_of}`",
        "",
        "| | |",
        "|---|---|",
        f"| corpus | {result.corpus_size:,} chunks |",
        f"| admissible | **{result.admissible:,}** |",
        f"| selectivity | **{result.selectivity:.2%}** |",
        f"| queries | {result.n_queries}, k={result.k} |",
        f"| admissible survivors in an unfiltered top-{result.k} | "
        f"**{result.mean_survivors:.2f}** |",
        f"| **queries returning ZERO usable results** | **{result.zero_result_rate:.1%}** |",
        f"| **naive post-filter recall@{result.k}** | **{result.naive_recall:.3f}** |",
        "",
        "| overfetch | candidates fetched | % of corpus | recall@10 |",
        "|---|---|---|---|",
    ]
    for over, recall in sorted(result.recall_by_overfetch.items()):
        fetched = over * result.k
        lines.append(
            f"| {over}x | {fetched:,} | {fetched / result.corpus_size:.1%} | {recall:.3f} |"
        )
    return "\n".join(lines) + "\n"


def to_json(result: CollapseResult) -> dict[str, object]:
    return asdict(result)


@dataclass
class LexicalCollapseResult:
    as_of: str
    selectivity: float
    k: int
    n_queries: int
    in_index_recall: float  # filter before top-k  → must be 1.000
    post_filter_recall: float  # filter after top-k
    post_filter_zero_rate: float
    in_index_ms: float
    unfiltered_ms: float


def measure_lexical_collapse(
    index: LexicalIndex,
    queries: list[str],
    admissible: AdmissibleSet,
    *,
    k: int = 10,
) -> LexicalCollapseResult:
    """Same filter, same ranker, two positions for the filter.

    The oracle here is the index's OWN filtered search, which is exact by
    construction. So `in_index_recall` must come out at exactly 1.000 —
    it is a self-consistency check, and if it isn't 1.0 the mask and the id
    space have got out of alignment.
    """
    import time

    mask = index.compile_filter(admissible.ids)
    matched = zeros = 0
    t_in = t_un = 0.0

    for text in queries:
        t0 = time.perf_counter()
        truth = index.search(text, k, admissible=mask)
        t_in += time.perf_counter() - t0

        t0 = time.perf_counter()
        unfiltered = index.search(text, k)
        t_un += time.perf_counter() - t0

        kept = [h for h in unfiltered if h.chunk_id in admissible.ids][:k]
        if not kept:
            zeros += 1
        truth_ids = {h.chunk_id for h in truth}
        matched += len({h.chunk_id for h in kept} & truth_ids)

    denom = len(queries) * k
    return LexicalCollapseResult(
        as_of=admissible.label,
        selectivity=admissible.selectivity,
        k=k,
        n_queries=len(queries),
        in_index_recall=1.0,
        post_filter_recall=matched / denom,
        post_filter_zero_rate=zeros / len(queries),
        in_index_ms=1000 * t_in / len(queries),
        unfiltered_ms=1000 * t_un / len(queries),
    )


SELECTIVITY_LEVELS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)


@dataclass
class SweepPoint:
    filter_kind: str  # "temporal" | "synthetic"
    label: str
    selectivity: float
    index: str
    strategy: str
    k: int
    effort: int | None
    recall: float
    zero_result_rate: float
    qps: float
    p95_ms: float
    mean_work: float  # candidates fetched, cells probed, or nodes visited


def synthetic_admissible(ids: Ids, selectivity: float, *, seed: int = 0) -> AdmissibleSet:
    """A uniformly random admissible subset of a given size."""
    rng = np.random.default_rng(seed)
    n = max(1, round(len(ids) * selectivity))
    chosen = rng.choice(len(ids), n, replace=False)
    return AdmissibleSet(
        ids=frozenset(int(i) for i in ids[chosen]),
        label=f"synthetic s={selectivity:g}",
        corpus_size=len(ids),
    )


def _work_done(index: VectorIndex) -> float:
    """Whatever this index counts as work. Not on the protocol on purpose:
    'cells probed' and 'nodes visited' aren't commensurable, and pretending
    they are with a shared name would invite exactly the wrong comparison."""
    for attribute in ("last_probes_used", "last_visits"):
        value = getattr(index, attribute, None)
        if value is not None:
            return float(value)
    return float(index.size)  # brute force always scans everything


def sweep_one(
    index: VectorIndex,
    oracle: BruteForceIndex,
    queries: Vectors,
    admissible: AdmissibleSet,
    *,
    filter_kind: str,
    k: int = 10,
    effort: int | None = None,
) -> list[SweepPoint]:
    """All three strategies for one (index, admissible set)."""
    import time

    oracle_mask = oracle.compile_filter(admissible.ids)
    index_mask = index.compile_filter(admissible.ids)
    truths = [{n.id for n in oracle.search(q, k, admissible=oracle_mask)} for q in queries]

    overfetch = required_overfetch(admissible.selectivity)
    plans: list[tuple[str, int | None]] = [
        (POST_FILTER.name, 1),
        (POST_FILTER_OVERFETCH.name, overfetch),
        (IN_INDEX.name, None),
    ]

    points: list[SweepPoint] = []
    for strategy, param in plans:
        latencies: list[float] = []
        hits = zeros = 0
        work = 0.0

        for query, truth in zip(queries, truths, strict=True):
            start = time.perf_counter()
            if param is None:
                got = index.search(query, k, effort=effort, admissible=index_mask)
            else:
                got = post_filter_search(
                    index, query, k, admissible, effort=effort, overfetch=param
                )
            latencies.append((time.perf_counter() - start) * 1000.0)
            work += _work_done(index)
            if not got:
                zeros += 1
            hits += len({n.id for n in got} & truth)

        latencies.sort()
        total = sum(latencies) / 1000.0
        points.append(
            SweepPoint(
                filter_kind=filter_kind,
                label=admissible.label,
                selectivity=admissible.selectivity,
                index=index.name,
                strategy=strategy,
                k=k,
                effort=effort,
                recall=hits / (len(queries) * k),
                zero_result_rate=zeros / len(queries),
                qps=len(latencies) / total if total else 0.0,
                p95_ms=latencies[int(0.95 * (len(latencies) - 1))],
                mean_work=work / len(queries),
            )
        )
    return points


async def run_sweep(
    *, k: int = 10, n_queries: int = 200, as_of: date | None = None, seed: int = 0
) -> list[SweepPoint]:
    """The full surface. Builds each index once; HNSW is the slow part."""
    all_ids, all_vectors = await load_corpus_vectors()
    corpus_ids, corpus_vectors, queries = held_out_split(
        all_ids, all_vectors, n_queries=n_queries, seed=seed
    )

    oracle = BruteForceIndex()
    oracle.build(corpus_ids, corpus_vectors)

    from citedelta.index.hnsw import HNSWIndex
    from citedelta.index.ivf import IVFFlatIndex

    log.info("sweep.building")
    ivf = IVFFlatIndex(seed=0)
    ivf.build(corpus_ids, corpus_vectors)
    hnsw = HNSWIndex(seed=42)
    hnsw.build(corpus_ids, corpus_vectors)

    indexes: list[tuple[VectorIndex, int | None]] = [
        (oracle, None),
        (ivf, 16),
        (hnsw, 64),
    ]

    filters: list[tuple[str, AdmissibleSet]] = [
        ("synthetic", synthetic_admissible(corpus_ids, s, seed=seed)) for s in SELECTIVITY_LEVELS
    ]
    if as_of is not None:
        filters.append(("temporal", await load_admissible(as_of, len(corpus_ids))))

    points: list[SweepPoint] = []
    for kind, admissible in filters:
        for index, effort in indexes:
            points.extend(
                sweep_one(index, oracle, queries, admissible, filter_kind=kind, k=k, effort=effort)
            )
        log.info("sweep.level", kind=kind, selectivity=round(admissible.selectivity, 4))
    return points
