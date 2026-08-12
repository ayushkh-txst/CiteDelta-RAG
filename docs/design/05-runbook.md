# 05 — Runbook

## Deploy

```bash
fly deploy
fly ssh console -C "alembic upgrade head"
curl -sf https://<app>.fly.dev/healthz
```

## Roll back

```bash
fly releases                       # find the last good version
fly deploy --image <previous-image>
```

**Migrations do not roll back automatically.** Every migration in this repo
has a working `downgrade()`, but check whether the previous release can read
the current schema before rolling back — additive migrations are safe,
destructive ones are not.

## Backup

```bash
pg_dump "$DATABASE_URL" -Fc -f citedelta-$(date +%F).dump
```

The corpus is ~38k chunks plus 79 snapshots; the dump is ~77 MB. **Index files
are not backed up** — they are derived, and rebuilding is faster than
restoring them (measured below).

## Restore

All commands below pin an explicit `DATABASE_URL`. Do not rely on defaults:
`createdb`/`pg_restore` with no URL target the default local server, not the
container — see "Went wrong" below.

```bash
# dump lives on a host that can reach the target server
createdb "$DATABASE_URL" citedelta_restore
pg_restore -d "$DATABASE_URL" --no-owner citedelta-YYYY-MM-DD.dump

# rebuild derived indexes, then serve
DATABASE_URL="$DATABASE_URL" uv run citedelta index build
DATABASE_URL="$DATABASE_URL" uv run citedelta serve --port 8001 &
```

If the target server is the local container `citedelta-pg` (port 5434), run the
create/restore inside the container instead:

```bash
docker exec -i citedelta-pg createdb -U citedelta citedelta_restore
docker exec -i citedelta-pg pg_restore -U citedelta -d citedelta_restore --no-owner < citedelta-YYYY-MM-DD.dump
```

The serving path needs three things: the schema + corpus + `embeddings` table
(restored above), and `data/index/lexical.idx` (`index build`). Vectors are
rebuilt in memory from the `embeddings` table at startup — there is no separate
vector-index build step in the serving path.

### Restore verified

| | |
|---|---|
| **Date performed** | 2026-08-12 |
| `pg_dump` (custom format) | ~77 MB |
| `pg_restore` | 5.9 s |
| `index build` | 2.6 s |
| First successful `/healthz` | 2 s |
| **Total** (restore → serving) | **~10.5 s** |
| Went wrong | see below |

**What went wrong:** the first attempt ran `createdb citedelta_restore` and
`pg_restore -d citedelta_restore` with no connection URL. Both hit the local
Postgres.app instance on port 5432, while the app talks to the container on
port 5434 — so the restore landed on the wrong server and the app failed with
`InvalidCatalogNameError: database "citedelta_restore" does not exist`. The
symptom was confusing because both commands *succeeded*. Fix: never restore
without an explicit `DATABASE_URL`, or run inside the container (above).
The timed numbers above are from the corrected run.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `/healthz` 500 on boot | Index files missing | `citedelta index build` |
| Every answer refuses | `anthropic_api_key` unset | Check the secret; retrieval still works without it |
| p95 climbing over time | Connection pool exhausted | `SELECT count(*) FROM pg_stat_activity` |
| `UnknownRate` on `/ask` | Model id has no rate row | Add a `Rate` to `pricing.py` |
| Answers ignore as-of | Index rebuilt without the new corpus | Rebuild after any ingest |
| `/ask` returns `502` | LLM provider down/auth | The outage is separate from refusals by design; check the provider |
| `InvalidCatalogNameError` after "successful" restore | Restored to the wrong server (no explicit URL) | Pin `DATABASE_URL` on every restore command |
| `struct.error` at boot | Corrupt `lexical.idx` | Restore the previous index file, then re-run `index build` (atomic-rename write protects against build-time crashes, not post-hoc corruption) |
