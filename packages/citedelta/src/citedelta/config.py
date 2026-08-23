"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import ssl
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every knob the app has. Anything not here is a hard-coded constant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://citedelta:citedelta@localhost:5434/citedelta"
    data_dir: Path = Path("./data")
    log_level: str = "info"

    llm_provider: str = "openrouter"
    """'openrouter' or 'anthropic'. Selects the adapter built in api/state.py
    and cli.py via substrate.llm.factory.build_completions — see
    docs/design/06-decisions/ADR-0023.md."""

    anthropic_api_key: str = ""
    openrouter_api_key: str = ""

    llm_model: str = "google/gemma-4-26b-a4b-it:free"
    resolver_model: str = "google/gemma-4-26b-a4b-it:free"
    """Deliberately not the answer model — see answer/resolve.py. Kept as its
    own setting rather than derived from llm_model so a deployment can pick a
    cheaper resolver even when it upgrades the answer model."""

    # Verbatim quotes made answers longer: a two-citation answer now carries a
    # full clause per citation, and 2048 output tokens truncated the JSON mid
    # string. 4096 keeps the same answer quality while allowing the quotes.
    llm_max_tokens: int = 4096

    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int = 512
    """Matryoshka-truncated from the model's native 1536 — see
    embed/openrouter.py for why 512 was chosen (Postgres free-tier budget)."""

    @property
    def sqlalchemy_url(self) -> str:
        """Alembic runs through SQLAlchemy, which wants the driver named in
        the URL. `sslmode` is stripped, not left in: SQLAlchemy's asyncpg
        dialect forwards unrecognized query params straight through as
        keyword arguments to `asyncpg.connect()`, which has no `sslmode`
        parameter — only `ssl`, as a real bool/SSLContext, not a query-string
        value. Confirmed live against Supabase: left in, this is a hard
        TypeError before any connection is attempted. See
        `sqlalchemy_connect_args` for how TLS is actually requested."""
        url = self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        parts = urlsplit(url)
        kept = [(k, v) for k, v in parse_qsl(parts.query) if k != "sslmode"]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))

    @property
    def sqlalchemy_connect_args(self) -> dict[str, Any]:
        """Extra asyncpg-level connect kwargs the URL alone can't carry —
        pass to `create_async_engine`/`async_engine_from_config`'s
        `connect_args`, alongside `sqlalchemy_url`.

        `statement_cache_size=0` unconditionally: a transaction/statement-
        mode PgBouncer (Supabase's pooler, among others) hands a physical
        connection to a different logical client between statements, so
        asyncpg's default prepared-statement cache collides across clients —
        DuplicatePreparedStatementError, confirmed live against Supabase.
        Harmless against a direct connection, just without the (for a
        migration run, negligible) caching win.

        `ssl` only when `database_url` actually asked for it via `sslmode`,
        so local docker-compose Postgres (no TLS configured) isn't forced
        into a handshake it can't do. `sslmode=require`'s own contract is
        "encrypt, don't verify the chain" — `CERT_NONE` reproduces that,
        where a bare `ssl=True` would ask for full verification instead and
        fail against a managed Postgres using its own CA (also confirmed
        live against Supabase).
        """
        args: dict[str, Any] = {"statement_cache_size": 0}
        sslmode = dict(parse_qsl(urlsplit(self.database_url).query)).get("sslmode")
        if sslmode and sslmode != "disable":
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            args["ssl"] = ctx
        return args

    @property
    def raw_cache_dir(self) -> Path:
        """Where downloaded eCFR XML is cached, so re-runs never re-fetch."""
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        """Where hand-built index files live."""
        return self.data_dir / "index"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached so config is parsed once per process, not once per call site."""
    return Settings()
