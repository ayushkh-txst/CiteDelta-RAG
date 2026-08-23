"""baseline_vectors: 384 -> 512 dims (OpenRouter embeddings, ADR-0023)

Revises: c9dfd505d7e4
"""

from collections.abc import Sequence

from alembic import op

revision: str = "d3f8a1c5e9b2"
down_revision: str | Sequence[str] | None = "c9dfd505d7e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A 512-dim vector will not INSERT into a vector(384) column — pgvector
    # enforces the declared width. The table is a materialized, droppable
    # baseline (see b5e3f0c9d1a2's docstring), so truncating it here is the
    # correct move: it gets fully repopulated by `citedelta bench run`
    # against the corpus's current embedding model, same as any dimension
    # change would require.
    op.execute("TRUNCATE TABLE baseline_vectors")
    op.execute("ALTER TABLE baseline_vectors ALTER COLUMN embedding TYPE vector(512)")


def downgrade() -> None:
    op.execute("TRUNCATE TABLE baseline_vectors")
    op.execute("ALTER TABLE baseline_vectors ALTER COLUMN embedding TYPE vector(384)")
