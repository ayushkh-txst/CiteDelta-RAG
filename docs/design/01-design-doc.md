# CiteDelta — Design Document

## 1. Problem and non-goals

Retrieval systems have no opinion about whether a passage is still in force.
Cosine similarity happily returns a policy memo superseded three years ago, and
for an F-1 student asking about OPT unemployment limits a stale answer isn't a
bad demo — it's a status violation.

The general shape is **versioned truth**, and it recurs in tax, benefits, export
controls, HR policy, and clinical guidelines. Critically it is a *retrieval
engine* problem, not a prompt problem: correctness requires the temporal
predicate to be enforced **inside the index**, because post-filtering an
approximate result set silently destroys recall when the k nearest neighbours
are all inadmissible.

### Non-goals

- **Not legal advice.** Every answer carries a citation and an effective date;
  out-of-scope questions are refused rather than guessed at.
- **Not a general web-scale search engine.** Corpus is a bounded regulatory
  body, tens of thousands of chunks, single-node.
- **Not a vector database.** The indexes here exist to be built and measured.
  §9 states what would actually be deployed.
- **Not multi-tenant.** No per-user data, so no isolation requirements.
- **Not real-time.** Regulations change on the order of weeks. Ingestion
  latency of hours is fine, and that assumption buys a lot of simplicity.

## 2. Functional requirements

1. Ingest a versioned regulatory corpus, preserving **valid time** (in force)
   and **transaction time** (on record).
2. Answer `search(query, as_of, k)` where `as_of` is a point in *either*
   timeline.
3. Return only passages admissible at `as_of`, with a citation resolving to a
   specific paragraph.
4. Refuse, visibly, when confidence is below threshold.
5. Rebuild indexes from Postgres alone; index files are derived artifacts.

## 3. Non-functional requirements

| # | Requirement | Status |
|---|---|---|
| NFR-1 | p95 end-to-end query < 200 ms at 50 QPS, 4 vCPU | not yet measured |
| NFR-2 | Lexical search p95 < 10 ms at 38,211 chunks | **measured: p95 8.9 ms** |
| NFR-3 | Recall@10 ≥ 0.95 vs. the brute-force oracle | **measured: 1.000 (IVF-Flat nprobe=32; HNSW ef=32 on dedup). Under the 1.9% temporal filter: 0.949 (IVF), 0.948–0.995 (HNSW ef=64–256), 1.000 (brute)** |
| NFR-4 | Citation validity = 1.00 (a fabricated citation is a hard failure) | planned |
| NFR-5 | Ingestion resumes from worker loss with no duplication or loss | **proved** (`chaos/kill_worker_mid_ingest.py`) |
| NFR-6 | Full rebuild from empty database in < 10 min, cache warm | **measured: 2.8 s ingest + 2.5 s index** |

## 4. Capacity estimation

Corpus, as ingested (8 CFR Part 214, 2016-12-23 → today):

| Quantity | Value |
|---|---|
| Snapshot dates | 79 |
| Section versions | 147 (31 sections) |
| Chunks | 38,211 |
| Mean chunk length | 183 tokens / 1,147 chars |
| Raw XML cached on disk | 73 MB |

**Lexical index, measured:** 6.55 MB total, of which 6.01 MB is postings.
Varint + delta-gap encoding vs. fixed-width u32 pairs: **3.81×** smaller.
Vocabulary 4,872 terms; the dictionary is resident, the postings blob is mmap'd.
Search latency over the full corpus: **p50 5.2 ms · p95 8.9 ms · p99 9.0 ms**.

**Vector index, measured.** Embeddings are `BAAI/bge-small-en-v1.5`, 384-dim,
run locally via ONNX — chosen over a hosted API so the benchmark is
*reproducible*, and a hosted model that silently reversions makes recall
numbers incomparable across runs.

### Measured (this corpus)

| | |
|---|---|
| chunks | 38,211 |
| distinct texts | 1,761 (duplicate ratio 21.7×) |
| intrinsic dimensionality | 1.1 |
| float32 vectors, 384-dim | 58.7 MB |
| brute-force index | 58.2 MB |
| IVF-Flat (+ 194 centroids) | 58.5 MB |
| HNSW (922,562 edges × 4 B) | 62.0 MB |
| HNSW build time | 291 s |
| embedding wall time (cold / cached) | ~4.5 min / 0 min |

The dominant cost is the vectors themselves; the HNSW graph adds roughly 6% on
top of the raw vectors. Everything is resident on a 4 GB VPS with room to
spare, so the design may assume no paging.

### Projection to 10M chunks

float32 vectors alone: 10M × 384 × 4 = **15.4 GB** — past any cheap single
node. In order: int8 scalar quantization with a rescoring pass (3.8 GB), then
sharding by time range, which this corpus makes natural — a query already
names a date and most name a recent one. The partition key falls out of the
domain rather than being imposed on it.

## 5. Data model

Bitemporal at **section-version** granularity, because that is the grain at
which eCFR reports change. Attaching intervals to chunks would invent
precision the source doesn't have and require a stable per-paragraph identity
across rewrites, which paragraphs don't have (they get renumbered).

| index | why it exists |
|---|---|
| `sv_current_belief_uniq` (partial unique) | at most one current-belief version per (document, section, effective_from) |
| `sv_no_overlap` (GiST EXCLUDE, partial) | makes overlapping validity intervals structurally impossible; also serves as-of range containment queries |
| `chunks_citation_prefix_idx` (text_pattern_ops) | `citation_path LIKE '8 CFR 214.2(f)%'` narrowing |
| `chunks_sha_idx` | dedupe probe; the join key for the embedding cache |
| `embeddings` PK `(model_id, content_sha256)` | content-addressed vector cache; `model_id` in the key so switching models cannot serve stale vectors |

Chunks are immutable children of a section-version, so a change produces a new
version and a fresh set of chunks. Nothing is ever updated in place, which is
why the indexes never have to handle mutation.

## 6. API contract

`search(query, as_of, k)` where `as_of` names a point in **either** timeline:

- `valid_on` — when the rule was in force in the world
- `known_at` — restricted to what had been recorded by that instant (optional;
  omitting it means "use everything we know now")

**Semantics.** A result is returned only if its section-version's validity
interval contains `valid_on`, our belief was not superseded by `known_at`, and
the section is not a tombstone. Every result carries its citation path and its
effective range — an answer without an effective date is not a valid answer in
this domain. All three retrievers honour a single compiled admissibility filter,
so the temporal predicate is enforced identically across lexical and vector
search.

**Idempotency.** Retrieval is a pure read; identical inputs give identical
outputs, including tie-breaks (distance, then chunk id — enforced so that
benchmarks are reproducible and so pagination is stable).

**Error taxonomy.**

| condition | behaviour |
|---|---|
| `as_of` before the corpus horizon (2016-12-23) | empty result, not an error — we genuinely don't know |
| no admissible chunks | empty result with the admissible count reported |
| filter budget exhausted | best-effort results, flagged in the trace as degraded |
| `k` exceeds admissible count | fewer than k returned, **never** padded with inadmissible rows |

That last row is the one that matters in a compliance product: padding would
cite a rule that was not in force.

## 7. Failure modes

| what fails | detection | blast radius | recovery | automatic? |
|---|---|---|---|---|
| ingest worker dies mid-snapshot | job lease expires | one snapshot date | reclaimed by the ordinary claim query; writes were transactional | yes |
| poison snapshot crashes workers repeatedly | `attempts` hits `max_attempts` | one date | swept to DLQ; corpus stays consistent | yes |
| index file write interrupted | temp file never renamed | none — previous index still served | rebuild from Postgres | yes |
| embedding model changed | `model_id` mismatch → cache miss | none; re-embeds | re-run `embed run` | yes |
| filter budget exhausted on a hot query | trace flags degraded | that query's recall | raise budget or narrow the corpus | no — policy |
| overlapping validity intervals ingested | GiST EXCLUDE violation at insert | one transaction, rolled back | fix the timeline builder | yes |
| corpus horizon misread as "no rule" | — | **user-visible wrong answer** | documented limitation; UI must distinguish "not in force" from "not known" | **no** |

The last row is the honest one: our data starts in 2016-12-23, and "we have no
record" is not the same claim as "no rule existed." The system must never
conflate them, and today's `AsOf` semantics are what keep them separable.

## 8. Delivery semantics

Ingestion is an at-least-once scan over snapshot dates whose output is a
byte-identical corpus whatever the concurrency. Each worker leases a snapshot
date with a fencing token; writes are wrapped in transactions; on crash the
lease expires and the date is reclaimed. Snapshot submission is idempotent by
(`snapshot_date`), so a retried worker cannot double-apply a snapshot. Indexes
are derived artifacts rebuilt from Postgres alone — an index file that loses
the race is unlinked and the old one keeps serving.

## 9. Alternatives rejected

| alternative | why rejected |
|---|---|
| hosted embedding API | a reverted model silently invalidates every recall number; ONNX local is reproducible |
| post-filter the k-nearest | measured: recall 0.020 and 82% empty queries at 1.91% temporal selectivity ([benchmark](03-benchmarks.md#temporal-pushdown)) |
| post-filter with overfetch | correct, but the fetch multiplier is set by the query's date, scales as 1/selectivity ([benchmark](03-benchmarks.md#temporal-pushdown)) |
| prune inadmissible nodes from the HNSW walk | the admissible subgraph has mean degree 1.04 and shatters — pruning caps recall near 0.17 ([benchmark](03-benchmarks.md#temporal-pushdown)) |
| a fresh index per `as_of` date | 79 dates × 58 MB resident; every new snapshot rebuilds another; pushdown adds a ~1 ms per-request mask instead |
| a vector database | the point of the project is building the retrieval machinery and measuring it |
| score-normalized fusion | normalized scores destroy confidence information; RRF fuses on rank ([ADR-0014](06-decisions/ADR-0014.md)) |

## 10. Scaling to 100×

At 100× (3.8M chunks) the vector block is 15.4 GB float32 — past a cheap
single node, but under it with int8 quantization (3.8 GB) and a rescoring pass
over the ANN candidates. Beyond that, shard by time range: the partition key
falls out of the domain (a query names a date, usually a recent one) rather
than being imposed on it. Worker concurrency and the GiST constraint both scale
linearly with the ingest stream; the DLQ and lease machinery are already the
mechanism a distributed ingest would need. The temporal filter is orthogonal:
a date-scoped shard shrinks the admissible set, which only widen effective
selectivity and helps the filtered indexes.