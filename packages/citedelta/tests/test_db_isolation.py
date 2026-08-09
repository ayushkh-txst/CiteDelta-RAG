"""The test suite must not be able to reach the dev database."""

from __future__ import annotations

import conftest


def test_test_dsn_is_not_the_dev_dsn() -> None:
    assert conftest.TEST_DSN != conftest.DEV_DSN


def test_test_dsn_targets_a_test_database() -> None:
    assert conftest.TEST_DSN.rsplit("/", 1)[-1].endswith("_test")


def test_derivation_is_idempotent() -> None:
    """Deriving twice must not produce citedelta_test_test."""
    once = conftest._derive_test_dsn("postgresql://u:p@h:5434/citedelta")
    assert once.endswith("/citedelta_test")
    assert conftest._derive_test_dsn(once) == once


def test_derivation_preserves_credentials_and_port() -> None:
    out = conftest._derive_test_dsn("postgresql://u:p@h:5434/citedelta")
    assert out == "postgresql://u:p@h:5434/citedelta_test"
