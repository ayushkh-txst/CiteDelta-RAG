-- AS-OF query plan against the POPULATED corpus.
-- Compare with asof-empty.sql.
--
-- 147 section_versions / 38,211 chunks. The planner prefers a seq scan here:
-- at this size reading the whole table is genuinely faster than seeking the
-- GiST index (0.461 ms execution, 186 shared hits, all in the shared buffer).
-- With enable_seqscan = off the GiST path returns ~same time; not worth the
-- extra planning. At 100x the rows the exclusion-constraint GiST index is
-- there and will win — we never wrote a CREATE INDEX for it.

EXPLAIN (ANALYZE, BUFFERS)
SELECT c.id, c.citation_path, c.text
FROM chunks c
JOIN section_versions sv ON sv.id = c.section_version_id
WHERE sv.document_id = 1
  AND daterange(sv.effective_from, sv.effective_to, '[)') @> DATE '2019-06-01'
  AND sv.superseded_at IS NULL
  AND NOT sv.removed;

-- Forced index scan, for comparison:
SET enable_seqscan = off;
EXPLAIN (ANALYZE, BUFFERS)
SELECT c.id, c.citation_path, c.text
FROM chunks c
JOIN section_versions sv ON sv.id = c.section_version_id
WHERE sv.document_id = 1
  AND daterange(sv.effective_from, sv.effective_to, '[)') @> DATE '2019-06-01'
  AND sv.superseded_at IS NULL
  AND NOT sv.removed;
SET enable_seqscan = on;
