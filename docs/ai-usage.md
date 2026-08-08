# AI usage

This project was built with heavy AI assistance. That is the correct 2026
workflow; concealing it would be the problem, not the use.

## How it was used

- Step-by-step implementation guides, written against APIs probed live rather
  than described from memory.
- Draft implementations of the queue, parser, index, and bitemporal layer,
  which I then read line by line and modified.
- Rubber-ducking design trade-offs (queue substrate, DLQ shape, chunk grain,
  where to enforce the temporal predicate).

## What was rejected and why

- **A first parser that mis-handled CFR's em-dash designators** produced
  citations like `8 CFR 214.2(ii)(E)(2)…` with the top-level paragraph lost and
  runaway depth. Caught by running it against the real corpus and reading the
  output — this is why the designator parser was probed against real
  `2026-08-01` XML before being committed.
- **Hosted embedding API as the default** — kept local ONNX
  (`bge-small-en-v1.5`) instead, so the recall benchmark is reproducible across
  sessions; the hosted adapter exists only to prove the `EmbeddingProvider`
  port is real.

## The bar I hold myself to

I can whiteboard every design decision here cold, without notes: why
`SKIP LOCKED`, why `attempts` increments at claim, why full jitter, what `b`
does in BM25, why validity intervals are half-open `[from, to)`, and why
post-filtering an ANN result set destroys recall.