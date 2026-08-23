# Free hosting: Render + Supabase

No credit card on either side. See [ADR-0023](../design/06-decisions/ADR-0023.md)
for why OpenRouter is the provider that makes this possible.

**Not Hugging Face Spaces.** This doc originally targeted Spaces, but as of
August 2026 HF changed Docker Spaces to require a PRO (personal) or
Team/Enterprise (org) plan — free accounts can only create Static or Gradio
Spaces. Neither fits this app (FastAPI + Jinja templates + SSE streaming +
Postgres), so the app host below is **Render** instead. The `Dockerfile` and
`entrypoint.sh` at the repo root were written host-agnostic (they only need
`$PORT` and a couple of env vars), so nothing about the app itself changes —
only where it runs.

## 1. Database — Supabase

1. Create a free project at supabase.com (no card required).
2. In the SQL editor: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Copy the **pooler** connection string (not the direct one — the pooler is
   what survives a serverless/scale-to-zero host), and add `?sslmode=require`.
4. Budget check: 38k chunks × 512 dims × 4 bytes ≈ 78MB of vectors. Comfortable
   inside the free 500MB. The full 1536-dim model would have been ~233MB —
   this is why embeddings are truncated (see `embed/openrouter.py`).

## 2. Re-embed the corpus locally, then restore it to Supabase

The existing corpus (if any) was embedded with a different model/dimension —
`embeddings` is keyed by `(model_id, content_sha256)`, so the new vectors
don't overwrite the old ones. Re-embed against a **local** Postgres first
(faster than running the OpenRouter calls from inside the eventual host), then
move the data over:

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run alembic upgrade head          # picks up the 512-dim baseline migration
uv run citedelta embed run           # now hits OpenRouter, not fastembed

# Point at Supabase and load the corpus + embeddings:
pg_dump "$LOCAL_DATABASE_URL" --data-only --table=sources --table=documents \
  --table=section_versions --table=chunks --table=embeddings \
  | psql "$SUPABASE_DATABASE_URL"
```

Prune any leftover rows from a previous embedding model before dumping if
you're tight on the 500MB budget:
`DELETE FROM embeddings WHERE model_id <> 'openai/text-embedding-3-small@512';`

## 3. App — Render

No card required for Render's free web-service tier. Docker is a first-class
runtime (unlike Spaces, no plan gate).

1. New → Web Service → connect this GitHub repo. Render auto-detects the root
   `Dockerfile`; no `render.yaml` is required, though one is included at the
   repo root for a one-click Blueprint deploy (`New → Blueprint`).
2. Plan: **Free**. Region: whichever is closest to your Supabase project (less
   round-trip latency per query).
3. Environment variables (Render dashboard → Environment, or filled in
   automatically if using the Blueprint):
   - `OPENROUTER_API_KEY`
   - `DATABASE_URL` — the Supabase pooler URI from step 1
   - `LLM_PROVIDER=openrouter` (matches the code default; explicit here so the
     dashboard shows it)
4. Deploy. `entrypoint.sh` runs on every boot: `alembic upgrade head`, rebuilds
   the git-ignored lexical index from Postgres (`citedelta index build` —
   Render's disk is ephemeral across deploys/restarts), then serves on
   `$PORT` (Render injects this; falls back to `7860` if unset, e.g. for a
   local `docker run`).

## Known constraints

- **Cold starts.** Render's free web services sleep after 15 minutes of
  inactivity and take 30–60s to wake — on top of that, this app's own startup
  re-runs `citedelta index build` and reloads ~38k vectors into memory before
  it can serve, so the first request after a sleep will be noticeably slower
  than a typical cold start. Not something the app can hide; worth setting
  expectations for a demo link.
- **Single worker.** `api/app.py`'s in-flight-turn state (`_PENDING`) is
  process-global by design — do not add `--workers` to the serve command or
  scale to multiple replicas without also solving that.
- **Free-tier rate limits.** `google/gemma-4-26b-a4b-it:free` allows 50
  requests/day per OpenRouter account (20/min) until you've ever purchased
  $10 of credit, after which it's 1,000/day permanently. Each turn makes two
  calls (resolver + answer), so the untouched free tier caps a live demo at
  roughly 25 turns/day — and that budget is shared with anything else using
  the same OpenRouter key (see "reusing these credentials for a second app"
  below).
- **Answer quality.** The 26B free model is well below `claude-opus-5`; the
  citation validator will refuse more often. Worth measuring against the
  existing recall benchmark rather than assuming it's fine.

## Reusing these credentials for a second app

- **`OPENROUTER_API_KEY`** — safe to reuse as-is. It's just an API key, not
  tied to one app. The one thing to know: the 50/1,000-requests-per-day free
  cap on `:free` models is per OpenRouter **account**, not per app — two apps
  sharing this key draw from the same daily budget.
- **`SUPABASE_DATABASE_URL`** — don't point a second, unrelated app at the
  *same* database/schema as this one; a migration or table name collision in
  either app can break the other silently. Supabase's free tier allows
  multiple projects per account, so create a second free project for the
  other app instead of sharing this one.
- **`HF_TOKEN`** — not used by this deployment at all (Docker Spaces require
  PRO, and this app doesn't fit Gradio/Static). It's only relevant to the
  second app if that one can run as a Gradio or Static Space.
