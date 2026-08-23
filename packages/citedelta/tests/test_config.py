"""Settings' translation of DATABASE_URL for the asyncpg-via-SQLAlchemy path
(Alembic only — everywhere else talks to Postgres through raw asyncpg).
Two real bugs live here, both only visible against a pooled/managed
Postgres, not the local docker-compose one:

1. SQLAlchemy's asyncpg dialect forwards unrecognized query params straight
   through as kwargs to `asyncpg.connect()`. `sslmode` (libpq/psycopg-style,
   what every managed Postgres hands out) isn't a parameter asyncpg's
   `connect()` accepts — only `ssl`, and only a real bool/SSLContext, not a
   query-string value. Left in the URL, this is a hard TypeError.
2. A transaction/statement-mode PgBouncer (Supabase's pooler, among others)
   hands a physical connection to a different logical client between
   statements, so asyncpg's default prepared-statement cache collides
   across clients — DuplicatePreparedStatementError. Both confirmed live
   against Supabase's transaction pooler before this fix existed.
"""

from __future__ import annotations

import ssl

from citedelta.config import Settings


def _settings(database_url: str) -> Settings:
    return Settings(database_url=database_url)


def test_sqlalchemy_url_swaps_the_driver() -> None:
    s = _settings("postgresql://u:p@host:5432/db")
    assert s.sqlalchemy_url == "postgresql+asyncpg://u:p@host:5432/db"


def test_sqlalchemy_url_strips_sslmode() -> None:
    """Left in, SQLAlchemy forwards it as a kwarg asyncpg.connect() doesn't
    have, raising TypeError before any connection is attempted."""
    s = _settings("postgresql://u:p@host:5432/db?sslmode=require")
    assert "sslmode" not in s.sqlalchemy_url
    assert s.sqlalchemy_url == "postgresql+asyncpg://u:p@host:5432/db"


def test_sqlalchemy_url_preserves_other_query_params() -> None:
    s = _settings("postgresql://u:p@host:5432/db?sslmode=require&application_name=citedelta")
    assert "application_name=citedelta" in s.sqlalchemy_url
    assert "sslmode" not in s.sqlalchemy_url


def test_connect_args_always_disables_the_statement_cache() -> None:
    s = _settings("postgresql://u:p@host:5432/db")
    assert s.sqlalchemy_connect_args["statement_cache_size"] == 0


def test_connect_args_has_no_ssl_when_url_does_not_request_it() -> None:
    """Local docker-compose Postgres has no TLS configured — forcing a
    handshake it can't do would break every local `alembic upgrade head`."""
    s = _settings("postgresql://citedelta:citedelta@localhost:5434/citedelta")
    assert "ssl" not in s.sqlalchemy_connect_args


def test_connect_args_disables_ssl_when_url_says_disable() -> None:
    s = _settings("postgresql://u:p@host:5432/db?sslmode=disable")
    assert "ssl" not in s.sqlalchemy_connect_args


def test_connect_args_builds_an_unverified_ssl_context_for_require() -> None:
    """sslmode=require's own contract is 'encrypt, don't verify the chain' —
    a bare `ssl=True` asks asyncpg for full certificate verification
    instead, which is stricter than what was requested and breaks against a
    managed Postgres using its own CA (confirmed live against Supabase)."""
    s = _settings("postgresql://u:p@host:5432/db?sslmode=require")
    ctx = s.sqlalchemy_connect_args["ssl"]
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False
