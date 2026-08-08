"""corpus: bitemporal sections and chunks

Revision ID: 6e4bc42d9a10
Revises:
Create Date: 2026-08-08 22:15:08.696061

"""

from collections.abc import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "6e4bc42d9a10"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    op.execute("""
        CREATE TABLE sources (
            id         SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            slug       TEXT        NOT NULL UNIQUE,
            name       TEXT        NOT NULL,
            base_url   TEXT        NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE documents (
            id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            source_id   SMALLINT    NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            external_id TEXT        NOT NULL,   -- 'title-8/part-214'
            title       TEXT        NOT NULL,   -- 'Nonimmigrant Classes'
            citation    TEXT        NOT NULL,   -- '8 CFR Part 214'
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (source_id, external_id)
        )
    """)

    op.execute("""
        CREATE TABLE section_versions (
            id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            document_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            section        TEXT    NOT NULL,          -- '214.2'
            heading        TEXT    NOT NULL DEFAULT '',

            -- VALID TIME: when the rule was in force in the world.
            effective_from DATE    NOT NULL,
            effective_to   DATE,                      -- NULL = still in force

            -- TRANSACTION TIME: when the fact entered the record.
            issue_date     DATE        NOT NULL,      -- eCFR's issue_date
            recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            superseded_at  TIMESTAMPTZ,               -- NULL = current belief

            removed        BOOLEAN NOT NULL DEFAULT FALSE,  -- tombstone
            content_sha256 BYTEA   NOT NULL,

            CONSTRAINT sv_interval_sane
                CHECK (effective_to IS NULL OR effective_to > effective_from),
            CONSTRAINT sv_sha_is_sha256
                CHECK (octet_length(content_sha256) = 32)
        )
    """)

    # at most one current-belief version per start date
    op.execute("""
        CREATE UNIQUE INDEX sv_current_belief_uniq
            ON section_versions (document_id, section, effective_from)
            WHERE superseded_at IS NULL
    """)

    # no two current-belief versions of a section may overlap in time
    op.execute("""
        ALTER TABLE section_versions
            ADD CONSTRAINT sv_no_overlap
            EXCLUDE USING gist (
                document_id WITH =,
                section     WITH =,
                (daterange(effective_from, effective_to, '[)')) WITH &&
            ) WHERE (superseded_at IS NULL)
    """)

    op.execute("""
        CREATE TABLE chunks (
            id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            section_version_id BIGINT  NOT NULL
                                   REFERENCES section_versions(id) ON DELETE CASCADE,
            ordinal            INTEGER NOT NULL,   -- position within the section
            citation_path      TEXT    NOT NULL,   -- '8 CFR 214.2(f)(10)(ii)(A)'
            text               TEXT    NOT NULL,
            char_count         INTEGER NOT NULL,
            token_count        INTEGER NOT NULL,
            content_sha256     BYTEA   NOT NULL,
            UNIQUE (section_version_id, ordinal)
        )
    """)

    # citation LIKE '8 CFR 214.2(f)%' needs text_pattern_ops
    op.execute("""
        CREATE INDEX chunks_citation_prefix_idx
            ON chunks (citation_path text_pattern_ops)
    """)

    op.execute("CREATE INDEX chunks_sha_idx ON chunks (content_sha256)")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS section_versions")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS sources")
    # btree_gist stays; dropping it could break other schemas in this database.
