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
| NFR-3 | Recall@10 ≥ 0.95 vs. the brute-force oracle | **measured: 1.000 (IVF-Flat nprobe=32; HNSW ef=32 on dedup)** |
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

## 5–10

To be written: data model rationale, API contract, failure modes, delivery
semantics, alternatives rejected, scaling to 100×.