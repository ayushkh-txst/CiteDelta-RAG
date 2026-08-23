# Free-tier deployment image (Hugging Face Spaces, Render, or anywhere else
# that runs an arbitrary Docker container). No fastembed/onnxruntime here —
# embeddings go through OpenRouter (ADR-0023), which is what keeps this image
# small enough to fit a free host's build/runtime budget.
FROM python:3.12-slim

RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

# HF Spaces (and most container platforms) run as an arbitrary non-root uid
# unless told otherwise — create a real user rather than relying on that.
RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app/citedelta
ENV PATH="/home/app/.local/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT="/home/app/citedelta/.venv" \
    PYTHONUNBUFFERED=1

COPY --chown=app:app pyproject.toml uv.lock .python-version ./
COPY --chown=app:app packages/substrate/pyproject.toml packages/substrate/pyproject.toml
COPY --chown=app:app packages/citedelta/pyproject.toml packages/citedelta/pyproject.toml

# Split from the full COPY below so dependency install is cached across
# source-only changes — a rebuild after editing app code doesn't re-resolve.
RUN mkdir -p packages/substrate/src/substrate packages/citedelta/src/citedelta \
    && uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app packages/substrate packages/substrate
COPY --chown=app:app packages/citedelta packages/citedelta
COPY --chown=app:app alembic alembic
COPY --chown=app:app alembic.ini alembic.ini
COPY --chown=app:app entrypoint.sh entrypoint.sh

RUN uv sync --frozen --no-dev

# Set only after the build's own `uv sync` calls: `uv run` auto-resyncs by
# default, and would otherwise try to pull the dev dependency group (mypy,
# matplotlib, hnswlib — the last needs a C++ toolchain this image doesn't
# have) on every container start. The venv built above is already correct.
ENV UV_NO_SYNC=1

EXPOSE 7860
ENTRYPOINT ["./entrypoint.sh"]
