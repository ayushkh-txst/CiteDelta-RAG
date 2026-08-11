"""query traces

Revises: f017ed8dc228
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a120a07d6f0f"
down_revision: str | Sequence[str] | None = "f017ed8dc228"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE query_traces (
            id                 BIGSERIAL PRIMARY KEY,
            run_id             TEXT        NOT NULL,
            query              TEXT        NOT NULL,
            as_of              DATE        NOT NULL,

            -- Retrieval shape at the moment of the query. Selectivity in
            -- particular is not reconstructible later: it depends on the
            -- corpus as it stood, and the corpus grows.
            selectivity        DOUBLE PRECISION NOT NULL,
            candidates_lexical INTEGER     NOT NULL,
            candidates_vector  INTEGER     NOT NULL,

            -- Every fused candidate, INCLUDING the ones that were retrieved
            -- and then not cited. Those are the rows that make this a trace
            -- rather than a result list: "BM25 ranked this 3rd, the vector
            -- index never returned it, and the answer didn't use it" is the
            -- interesting sentence, and it is unrecoverable after the fact.
            candidates         JSONB       NOT NULL,

            cited_ids          BIGINT[]    NOT NULL DEFAULT '{}',
            answer             TEXT,
            refusal_reason     TEXT,
            refusal_detail     TEXT,

            latency_ms         DOUBLE PRECISION NOT NULL,
            cost_usd           NUMERIC(12, 6)   NOT NULL DEFAULT 0,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

            -- Exactly one outcome. The answer path has two terminal states
            -- and the database says so, rather than trusting every writer to
            -- remember. Same instinct as the GiST EXCLUDE constraint on
            -- section_versions: if an invariant is checkable in Postgres,
            -- check it in Postgres.
            CONSTRAINT query_traces_answer_xor_refusal
                CHECK ((answer IS NOT NULL) <> (refusal_reason IS NOT NULL))
        )
    """)
    op.execute("CREATE INDEX query_traces_created_at_idx ON query_traces (created_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS query_traces")
