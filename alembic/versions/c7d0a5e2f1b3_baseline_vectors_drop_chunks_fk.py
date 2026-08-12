"""baseline_vectors: drop the chunks FK

Revises: b5e3f0c9d1a2
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c7d0a5e2f1b3"
down_revision: str | Sequence[str] | None = "b5e3f0c9d1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The benchmark holds 500 vectors out as queries and may run synthetic
    # datasets whose ids are not chunk ids. The baseline table must accept
    # whatever dataset the harness hands it; the FK to `chunks` was a
    # convenience, not a requirement, and it makes `--dataset all` crash on
    # random-hard. The table is still droppable without touching the corpus.
    op.execute(
        "ALTER TABLE baseline_vectors DROP CONSTRAINT IF EXISTS baseline_vectors_chunk_id_fkey"
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE baseline_vectors
        ADD CONSTRAINT baseline_vectors_chunk_id_fkey
        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        """
    )
