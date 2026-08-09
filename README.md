# CiteDelta

Time-aware search over versioned regulations. Every answer carries an effective
date and a citation to the CFR.

Ask for §214.1 as of 2019-06-01 and you get the 2019 text. The temporal
predicate is part of the index, not a filter bolted on afterwards.

## Background

The hard part isn't the search, it's the dates. 8 CFR §214.1 was amended
effective 2017-01-18, but eCFR didn't record it until 2018-12-22. For eleven
months the official corpus served the old text — and correctly, because that was
the best available knowledge at the time. So "what was the rule on 2018-01-01?"
has two defensible answers, and a compliance product needs both:

- **valid time** — when the rule was actually in force
- **transaction time** — when the fact entered the record

34 of the records in 8 CFR Part 214 carry this gap. This project models both.

## What's in here

The point of the project is building retrieval machinery instead of importing
it. Everything listed is written from scratch:

- Durable job queue on Postgres: `SKIP LOCKED`, leases, fencing tokens, DLQ
- Bitemporal schema with a GiST exclusion constraint
- Inverted index: varint postings, mmap, BM25
- Three ANN indexes — brute force, IVF-Flat, HNSW — behind one protocol
- Temporal predicate pushdown and post-filter recall measurement (next)

## Vector search

Three indexes share a protocol, a conformance suite, and a benchmark harness:

- **brute force** — exhaustive scan; the correctness oracle. Recall 1.000, 887 QPS.
- **IVF-Flat** — spherical k-means + inverted lists, `nprobe` dial. Recall 0.992
  at 24,120 QPS (nprobe=1), 1.000 at 934 QPS (nprobe=32).
- **HNSW** — hierarchical navigable small-world graph, `ef` dial. Recall 1.000
  on the deduplicated corpus at 2,221 QPS.

![recall vs QPS](docs/design/benchmarks/recall-vs-qps.png)

One finding worth stating plainly: this corpus is easy for ANN search. 38,211
chunks collapse to 1,761 distinct texts (21.7× repeats across versions), and the
intrinsic dimensionality is ≈1.1 in a 384-dim space, so recall hits near-perfect
at minimal effort. The benchmarks include a deliberately hard synthetic dataset
so the accuracy/speed tradeoff is visible at all. Methodology and full numbers:
[docs/design/03-benchmarks.md](docs/design/03-benchmarks.md).

## Numbers

- Corpus: 38,211 chunks, 147 section versions, 79 snapshot dates (2016→2026)
- Lexical index: 6.5 MB; varint + delta-gap encoding cuts postings 3.8×
- Search latency: p50 5.2 ms · p95 8.9 ms · p99 9.0 ms
- Embedded corpus: 1,761 distinct texts, 21.7× dedup, intrinsic dim 1.1
- ANN: brute 887 QPS / IVF-Flat 24,120 QPS @ 0.992 / HNSW 1.0 recall (dedup)
- Crash recovery: `SIGKILL` mid-ingest still yields a byte-identical corpus
  ([chaos test](chaos/kill_worker_mid_ingest.py))

## Running it

```bash
git clone https://github.com/ayushkh-txst/CiteDelta-RAG.git && cd CiteDelta-RAG
cp .env.example .env
make up && make sync
uv run alembic upgrade head
uv run citedelta plan && uv run citedelta work -c 2
uv run citedelta index build
uv run citedelta search "optional practical training stem extension"
```

## Docs

[Design doc](docs/design/01-design-doc.md) ·
[ADRs](docs/design/06-decisions/) ·
[Benchmarks](docs/design/03-benchmarks.md) ·
[AI usage](docs/ai-usage.md)

---
*Not legal advice. Regulatory text is reproduced from the public eCFR API.*