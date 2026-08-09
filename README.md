# CiteDelta

**Time-aware retrieval over versioned regulations.** Every answer carries an
effective date and a citation.

A rule changed on 2020-10-02. Ask what it said on 2019-06-01 and you get the
2019 text — because the temporal predicate lives inside the index, not in a
filter applied afterwards.

## Why this is hard

8 CFR §214.1 was **amended effective 2017-01-18** and **recorded by eCFR on
2018-12-22**. For eleven months the official corpus served the old text — and
was not wrong to, because that was the best available knowledge. So
"what was the rule on 2018-01-01?" has two correct answers, and a compliance
product needs both:

- **valid time** — when the rule was in force in the world
- **transaction time** — when the fact entered the record

34 records in 8 CFR Part 214 have this gap. CiteDelta models both.

## Built from scratch (not imported)

| Component | Status |
|---|---|
| Durable job queue on Postgres — `SKIP LOCKED`, leases, fencing tokens, DLQ | ✅ Built |
| Bitemporal storage with a GiST exclusion constraint | ✅ Built |
| Inverted index — varint postings, mmap, BM25 | ✅ Built |
| ANN indexes — brute force → IVF-Flat → HNSW | ✅ Built |
| Temporal predicate pushdown + measured recall collapse of post-filtering | Planned |
| RRF fusion, pgvector baseline, retrieval-trace inspector | Planned |

## Vector search, written from scratch

Three indexes behind one protocol, one conformance suite, one benchmark harness:

| index | what it is | recall@10 | QPS |
|---|---|---|---|
| brute force | exhaustive scan — the correctness oracle | 1.000 | 887 |
| IVF-Flat | spherical k-means + inverted lists, `nprobe` dial | 0.992 (nprobe=1) · 1.000 (nprobe=32) | 24,120 · 934 |
| HNSW | hierarchical navigable small-world graph, `ef_search` dial | 1.000 (ef=32, dedup) | 2,221 |

![recall vs QPS](docs/design/benchmarks/recall-vs-qps.png)

**The corpus has intrinsic dimensionality ≈1.1 in a 384-dim space** — CFR text
is formulaic and repeats across versions (21.7× duplicate vectors). So ANN
search hits near-perfect recall at minimal effort, and that says more about
the corpus than the index. The benchmark includes a deliberately hard
synthetic dataset so the accuracy/speed knobs have somewhere to show a real
tradeoff. [Full methodology](docs/design/03-benchmarks.md).

## Measured

| | |
|---|---|
| Corpus | 38,211 chunks across 147 section versions, 79 snapshot dates, 2016→2026 |
| Lexical index | 6.5 MB, 3.8× smaller with varint + delta-gap encoding (4,872 terms) |
| Search latency | p50 5.2 ms · p95 8.9 ms · p99 9.0 ms over the full corpus |
| Embedded corpus | 38,211 chunks → 1,761 distinct texts (21.7× dedup), intrinsic dim 1.1 |
| ANN indexes | brute 887 QPS / IVF-Flat 24k QPS @ 0.992 / HNSW 1.0 recall (dedup) |
| Crash recovery | `SIGKILL` mid-ingest → byte-identical corpus ([chaos test](chaos/kill_worker_mid_ingest.py)) |

## Run it

```bash
git clone https://github.com/ayushkh-txst/CiteDelta-RAG.git && cd CiteDelta-RAG
cp .env.example .env
make up && make sync
uv run alembic upgrade head
uv run citedelta plan && uv run citedelta work -c 2
uv run citedelta index build
uv run citedelta search "optional practical training stem extension"
```

## Design docs

[Design doc](docs/design/01-design-doc.md) ·
[ADRs](docs/design/06-decisions/) ·
[Benchmarks](docs/design/03-benchmarks.md) ·
[AI usage](docs/ai-usage.md)

---
*Not legal advice. Regulatory text is reproduced from the public eCFR API.*