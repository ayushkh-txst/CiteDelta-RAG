# 08 — Evaluation

60 cases across six classes. Every expectation was verified by reading the
regulation text in the corpus, not generated and assumed. Graders are
programmatic — no LLM judge, because a language model grading a language
model's citations shares the failure mode being tested.

## Results

| Metric | Value | Target |
|---|---|---|
| recall@5 | 0.883 | ≥ 0.80 |
| citation validity | **1.00** | **1.00** |
| refusal accuracy | 0.80 | ≥ 0.90 |
| temporal accuracy | **1.00** | **1.00** |

| Class | n | recall@5 | pass rate |
|---|---|---|---|
| factual | 15 | 0.93 | 1.00 |
| multi-hop | 10 | 0.90 | 0.90 |
| temporal | 10 | 0.70 | 0.90 |
| adversarial | 10 | 1.00 | 0.30 |
| ambiguous | 5 | 1.00 | 0.80 |
| deadline | 10 | 0.80 | 0.90 |

Overall pass rate **0.80** (48/60). p50 latency 7.1 s. Cost for the 60-case
run: **$1.81** (claude-opus-5; the block budget guessed $0.25 at $0.004/case —
the actual model is pricier, and the ledger shows the real number).

## Failure analysis

Retrieval misses (recall@5 contributes 7 of the 60 misses):

- **temp-02 / temp-09 (STEM extension, as_of 2019 / 2016).** Expected the
  17-month form at `8 CFR 214.16(c)`; top-5 instead returned the *current*
  24-month rule `214.2(f)(10)(ii)(C)`. This is exactly the temporal leak the
  class exists to catch: retrieval was admissible-aware, but the 17-month
  provision is a thin, older slab of text that the hybrid fuse did not rank
  high enough against the lexically rich 24-month rule.
- **deadline-05 / temp-07 (60-day grace, `214.1(l)(2)`).** Not in top-5. The
  provision covers E/H/L/O/TN grace, a rare co-occurrence of terms, and the
  query's wording ("grace period after ceasing employment") did not surface it.
- **fact-06 (full course of study, `214.2(f)(6)(i)(B)`).** A sibling chunk
  `214.2(f)(6)(i)(F)` was retrieved instead — same paragraph family, wrong
  sub-item. Chunk boundary, not a missing provision.
- **deadline-08 (5-month absence, `214.13(d)(8)`).** Not in top-5; the
  retrieval returned `214.2(f)(15)` and other OPT text instead.
- **multi-04 (reinstatement, `214.2(f)(16)(i)(A)` + `214.13(d)(7)`).** Neither
  retrieved; reinstatement text is spread across two sections and the fuse
  did not combine them.

Refusal failures (12 of 60 overall; 7 in adversarial alone):

- **adv-03, adv-04, adv-05, adv-06, adv-07, adv-09, adv-10.** All seven
  *answered* instead of refusing. Retrieval succeeded for each (there is
  plenty of OPT text), and the answer stage, finding admissible sources,
  produced a response rather than a refusal. adv-09 is instructive: asked
  for "a 30-day OPT unemployment limit", the system answered "there is no
  30-day limit; the applicable provision is `214.2(f)(10)(ii)(E)`" — a
  *correction* that is arguably the right behaviour, but the eval demands a
  refusal and the guarantee must be strict: an adversarial prompt should
  never be handed a grounded-looking answer. The refusal gate needs to fire
  on prompt intent, not only on retrieval failure.
- **ambig-03 ("What do I need to do to stay in status?").** Answered at
  length rather than refusing for being under-specified.
- **temp-10 (STEM requirements, as_of 2026-07-17).** One-off
  `malformed_response` — the provider's structured output was unusable. First
  run of the same case passed (trace 70); the rerun hit trace 130's bad
  response. Transient, not systemic.

The pattern worth stating plainly: the system's refusal is **retrieval
driven, not intent driven**. When admissible text exists, it answers —
including to prompts that demand an answer against the rules. That is the
headline weakness this eval found, and it is a product-design question, not a
retrieval one. No retrieval parameter was changed in response to any result.

## What this set does not measure

- **Answer quality.** Graders check that citations are real, retrieved, and in
  force. Whether the prose is a *good* summary is not measured — that needs
  human judgement at a scale this project doesn't have.
- **Coverage of 8 CFR.** 60 cases over a 38,211-chunk corpus. It is a
  smoke test with teeth, not a coverage claim.
- **Real user questions.** I wrote these. Actual users would ask worse ones,
  and the ambiguous class is my guess at how.
- **Fictional-subsection robustness.** adv-01 (`999.99`) and adv-08
  (`214.2(f)(99)`) both refused correctly, but both were caught by the
  *existence* check inside `validate_citations`, not by a hard refusal in the
  answer stage. Cases where the model invents a plausible *real* provision are
  a deeper threat than this set probes.
