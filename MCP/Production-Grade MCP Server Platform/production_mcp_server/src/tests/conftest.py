from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_SERVER__MEMORY__CHROMA_ENABLED", "false")
