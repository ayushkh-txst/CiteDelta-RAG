EXPLAIN (ANALYZE, BUFFERS)
SELECT c.id, c.citation_path, c.text
FROM chunks c
JOIN section_versions sv ON sv.id = c.section_version_id
WHERE sv.document_id = 1
  AND daterange(sv.effective_from, sv.effective_to, '[)') @> DATE '2019-06-01'
  AND sv.superseded_at IS NULL
  AND NOT sv.removed;

-- Ran against an empty corpus (Block 1). Tables are empty so the planner is
-- right to lean on the tiny indexes; see asof-populated.sql for the populated
-- version at the end of Block 4.

 Nested Loop  (cost=4.31..19.47 rows=1 width=72) (actual time=0.316..0.317 rows=1 loops=1)
   Buffers: shared hit=55 dirtied=2
   ->  Index Scan using sv_no_overlap on section_versions sv  (cost=0.14..8.16 rows=1 width=8) (actual time=0.301..0.302 rows=1 loops=1)
         Index Cond: ((document_id = 1) AND (daterange(effective_from, effective_to, '[)'::text) @> '2019-06-01'::date))
         Filter: (NOT removed)
         Buffers: shared hit=53 dirtied=1
   ->  Bitmap Heap Scan on chunks c  (cost=4.17..11.28 rows=3 width=80) (actual time=0.013..0.013 rows=1 loops=1)
         Recheck Cond: (section_version_id = sv.id)
         Heap Blocks: exact=1
         Buffers: shared hit=2 dirtied=1
         ->  Bitmap Index Scan on chunks_section_version_id_ordinal_key  (cost=0.00..4.17 rows=3 width=0) (actual time=0.006..0.006 rows=1 loops=1)
               Index Cond: (section_version_id = sv.id)
               Buffers: shared hit=1
 Planning:
   Buffers: shared hit=383
 Planning Time: 0.526 ms
 Execution Time: 0.347 ms
