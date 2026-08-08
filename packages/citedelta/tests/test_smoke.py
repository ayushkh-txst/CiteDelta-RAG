from citedelta import __version__
from citedelta.config import get_settings


def test_package_is_importable() -> None:
    assert __version__ == "0.1.0"


def test_settings_load() -> None:
    s = get_settings()
    assert s.database_url.startswith("postgresql://")
    assert s.sqlalchemy_url.startswith("postgresql+asyncpg://")


def test_substrate_is_importable_from_citedelta() -> None:
    """The dependency arrow points this way and only this way."""
    import substrate

    assert substrate.__version__
