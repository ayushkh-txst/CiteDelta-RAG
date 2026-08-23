#!/usr/bin/env bash
# Runs on every container start. The lexical index and the container's own
# filesystem are ephemeral on a free host (git-ignored on purpose, see
# .gitignore) — the source of truth is Postgres, so this rebuilds what's
# missing from it before serving.
set -euo pipefail

uv run alembic upgrade head
uv run citedelta index build

# Render injects $PORT; a bare local `docker run` won't set one, so fall
# back to 7860 (also what EXPOSE in the Dockerfile documents).
exec uv run citedelta serve --host 0.0.0.0 --port "${PORT:-7860}"
