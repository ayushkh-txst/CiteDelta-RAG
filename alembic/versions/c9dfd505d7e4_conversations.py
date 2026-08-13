"""conversations

Revision ID: c9dfd505d7e4
Revises: c7d0a5e2f1b3
Create Date: 2026-08-13 22:09:56.080198

"""

from collections.abc import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c9dfd505d7e4"
down_revision: str | Sequence[str] | None = "c7d0a5e2f1b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE query_traces
            ADD COLUMN conversation_id UUID,
            ADD COLUMN turn_index       INTEGER,
            -- What retrieval ACTUALLY ran. For a follow-up the user typed
            -- "what about then?" but we searched for something else entirely.
            -- Storing only the raw query would make every trace of a
            -- follow-up a lie about what happened.
            ADD COLUMN resolved_query   TEXT
        """
    )
    # Existing rows are single-turn conversations of their own. Backfilling
    # rather than leaving NULLs means the thread query needs no special case
    # for pre-existing history.
    op.execute(
        """
        UPDATE query_traces
           SET conversation_id = gen_random_uuid(),
               turn_index = 0
         WHERE conversation_id IS NULL
        """
    )
    op.execute(
        """
        ALTER TABLE query_traces
            ALTER COLUMN conversation_id SET NOT NULL,
            ALTER COLUMN turn_index      SET NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX query_traces_thread_idx
            ON query_traces (conversation_id, turn_index)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS query_traces_thread_idx")
    op.execute(
        """
        ALTER TABLE query_traces
            DROP COLUMN IF EXISTS resolved_query,
            DROP COLUMN IF EXISTS turn_index,
            DROP COLUMN IF EXISTS conversation_id
        """
    )
