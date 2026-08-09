"""embeddings content-addressed vector cache

Revision ID: f017ed8dc228
Revises: 94fa233b236b
Create Date: 2026-08-09 10:42:56.613706

"""

from collections.abc import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "f017ed8dc228"
down_revision: str | Sequence[str] | None = "94fa233b236b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embeddings (
            model_id       TEXT   NOT NULL,
            content_sha256 BYTEA  NOT NULL,
            dim            INTEGER NOT NULL,
            -- float32 little-endian. Deliberately NOT pgvector yet: today the
            -- vectors are read out to NumPy for indexes we wrote ourselves.
            -- The pgvector baseline will be measured on identical vectors.
            vector         BYTEA  NOT NULL,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

            PRIMARY KEY (model_id, content_sha256),
            CONSTRAINT embeddings_vector_len
                CHECK (octet_length(vector) = dim * 4)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embeddings")
