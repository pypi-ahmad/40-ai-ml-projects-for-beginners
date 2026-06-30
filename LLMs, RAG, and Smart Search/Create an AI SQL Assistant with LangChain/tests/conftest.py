"""Shared pytest fixtures for AI SQL assistant tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_sql_assistant.data.northwind import build_northwind_databases
from ai_sql_assistant.schema.introspector import inspect_database


@pytest.fixture(scope="session")
def sqlite_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build temporary raw/scaled test databases once per test session."""
    root = tmp_path_factory.mktemp("sqlite_data")
    raw = root / "northwind_raw.db"
    scaled = root / "northwind_scaled.db"
    build_northwind_databases(raw_db_path=raw, scaled_db_path=scaled, scale_factor=2, seed=11)
    return {"raw": raw, "scaled": scaled}


@pytest.fixture(scope="session")
def schema_report(sqlite_paths: dict[str, Path]) -> dict:
    """Introspected schema report on scaled test DB."""
    return inspect_database(sqlite_paths["scaled"])
