"""pgvector and tsvector baseline

Revises: a120a07d6f0f
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b5e3f0c9d1a2"
down_revision: str | Sequence[str] | None = "a120a07d6f0f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # A materialized copy rather than a column on `chunks`. Two reasons:
    #   1. The baseline must be droppable without touching the corpus schema.
    #   2. It keeps the comparison honest — this table holds EXACTLY the
    #      vectors the hand-written indexes were built from, so any
    #      difference in results is the index, not the data.
    op.execute("""
        CREATE TABLE baseline_vectors (
            chunk_id  BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            embedding vector(384) NOT NULL
        )
    """)

    # tsvector baseline for the lexical side. GIN, English config — the
    # default anyone would reach for, which is the point of a baseline.
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS ts tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
    """)
    op.execute("CREATE INDEX chunks_ts_idx ON chunks USING GIN (ts)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_ts_idx")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS ts")
    op.execute("DROP TABLE IF EXISTS baseline_vectors")
