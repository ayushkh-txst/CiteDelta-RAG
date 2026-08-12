# 04 — Failure injection

Each scenario names what was injected, what the system was supposed to do,
and what it actually did. Where those differ, the difference is the finding.

All scenarios run against the live service (`uv run citedelta serve`) and the
local Postgres container `citedelta-pg` (port 5434). Chaos scripts live in
`chaos/`.

## 1. Worker killed mid-job (SIGKILL)

**Injected:** `kill -9` on the queue worker mid-ingest (`chaos/kill_worker_mid_ingest.py`)
— the worker held a live lease and died with the corpus half-written.

**Expected:** the lease expires; another worker reclaims the job; no duplicate
side effects; the corpus is not half-written.

**Observed:** after killing the worker mid-write (~1,800 chunks in, 73 pending,
2 running), the runner restarted the worker, which reclaimed the in-flight jobs
and re-ingested the failed snapshot. Final corpus: **38,211 chunks, byte-identical
digest** to the pre-chaos corpus (same md5). **0 jobs dead, 0 jobs lost.**
Ingests are idempotent (keyed by snapshot date + `content_sha256`), so the
restart converges rather than duplicates.

**Mechanism:** visibility timeout + `lease_epoch` fencing. A resurrected worker
carrying a stale epoch cannot complete a job that has been reclaimed — its
UPDATE matches zero rows. The truncate-and-reingest path (`RESTART IDENTITY`)
plus `GENERATED` tsvector and content-keyed embeddings keeps the corpus
consistent after a re-ingest.

**Post-chaos checks:** `chunks.ts` is `GENERATED ALWAYS AS to_tsvector(...)
STORED` (repopulates on insert); `load_corpus_vectors` joins `embeddings` by
`content_sha256`; `query_traces` has no FK to `chunks`; `baseline_vectors`' FK
to `chunks` was dropped in migration `c7d0a5e2f1b3`, so the bench table survives
a truncate. Caveat: `baseline_vectors` holds pre-chaos chunk ids, so a pgvector
bench re-run needs `rebuild_baseline(probe=...)` first.

## 2. Postgres restarted under load

**Injected:** `docker restart citedelta-pg` while the server was serving.

**Expected:** in-flight queries fail, the pool reconnects, requests recover,
no jobs lost.

**Observed:** `/healthz` stayed green through the restart and for the full
24-second observation window after it — the asyncpg pool reconnected
transparently. `/search` returned the identical result set afterward
(5 hits, selectivity 0.0213). No request surfaced an error.

**Finding:** the serving path is pool-reconnect safe; this is the default
asyncpg behavior and worth one line in the runbook but not an alert. The
ingest path's restart was covered by scenario 1.

## 3. eCFR API returning 503

**Injected:** a local 503 proxy (`/tmp/ecfr_503_proxy.py`, port 8099) that
answers `503 Service Unavailable` to GET/CONNECT/POST; the worker's `httpx`
client was pointed at it via `HTTPS_PROXY`, so any eCFR fetch raised
`ProxyError('503 Service Unavailable')`. A synthetic ingest job
(`idempotency_key="injected-503@2030-01-01"`, `max_attempts=4`) was enqueued.

**Expected:** full-jitter retry, then DLQ after max attempts. Attempts are
incremented **at claim time**, not at failure time, so a job that crashes the
worker cannot retry forever.

**Observed:** the job cycled pending → running → pending with full-jitter
backoff (base 1.0 s, cap 30.0 s), hit `attempts = max_attempts = 4`, and
dead-lettered. Final state:

| | |
|---|---|
| job id | 80 |
| state | `dead` |
| attempts | 4 / 4 |
| `last_error` | `ProxyError('503 Service Unavailable')` |

Queue after the test: `{"pending":0,"running":0,"succeeded":79,"failed":0,"dead":1}`
(the synthetic job was then deleted). The dead-letter is visible in `jobs`
where `state='dead'`; a dead job is never auto-retried.

## 4. Corrupt index file

**Injected:** `truncate -s 100 data/index/lexical.idx` (postings file cut to
100 bytes) before starting the server.

**Expected:** a loud failure at load, not silently wrong results.

**Observed:**

```
struct.error: unpack_from requires a buffer of at least 305748 bytes
  for unpacking 305688 bytes at offset 60 (actual buffer size is 100)
ERROR:    Application startup failed. Exiting.
```

The server refused to start; `/healthz` never came up. A corrupt index is a
crash at boot, not a silent wrong-answer machine.

**Note:** index writes are atomic-rename, so a crash *during a build* leaves
the previous index intact. This scenario tested corruption after the fact,
which that design does not protect against — the load path validates sizes
but not a checksum. A torn file larger than the corruption checked here would
be caught by `struct.error`; a checksummed header is a future hardening step.

## 5. LLM provider unavailable

**Injected:** `ANTHROPIC_API_KEY=sk-ant-invalid-key-for-chaos-test` (a 401
`AuthenticationError`), then `POST /ask`.

**Expected:** `CompletionError` propagates as a **502**. **Not** a refusal —
an outage is not a decision, and rendering it as one would make the refusal
rate absorb the incident and the uptime graph stay green through it.

**Observed (first run):** the raw `anthropic.AuthenticationError` escaped the
adapter and surfaced as a bare **HTTP 500**. The design comment said "the API
layer turns it into a 502" — but no handler existed, and the adapter only
wrapped retryable errors into `CompletionError`; a 401 is not retryable, so it
bypassed the wrapper entirely.

**Finding and fix (two changes):**

1. `substrate/llm/anthropic_adapter.py` — non-retryable `anthropic.APIError`s
   (auth, invalid request, …) now raise `CompletionError` immediately instead
   of escaping raw. Any provider failure is a `CompletionError`; there is no
   third kind of provider error.
2. `api/app.py` — `create_app` registers a `CompletionError` exception handler
   that returns **502** with `{"detail": "model provider unavailable"}`,
   closing the loop the comment promised.

**Observed (after fix):**

```
{"detail": "model provider unavailable"}
HTTP 502
```

With the fix, an LLM outage is separable from both a refusal (200 + refusal
reason) and a code bug (500) in one status code.

## What is not covered

- Disk-full during index build (atomic rename leaves the previous index, but a
  full filesystem mid-write is untested)
- Postgres failover (single instance by design)
- Clock skew affecting `as_of` at a version boundary
- Corrupt `hnsw.npz`/vector artifact (serving path builds vectors from the
  `embeddings` table, not from the file; a torn `hnsw.npz` only affects the
  bench commands)
