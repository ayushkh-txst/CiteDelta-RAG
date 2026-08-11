"""Runtime configuration, read once from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 2048

    @property
    def sqlalchemy_url(self) -> str:
        """Alembic runs through SQLAlchemy, which wants the driver named in the URL."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

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
