from __future__ import annotations

from pathlib import Path

from config.settings import load_settings


def test_load_settings_from_yaml() -> None:
    settings = load_settings(Path("configs/default.yaml"))
    assert settings.app.name == "production-mcp-server"
    assert settings.transport.mode == "stdio"
    assert "qwen3:8b" in settings.models.supported_models


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MCP_SERVER__TRANSPORT__MODE", "http")
    settings = load_settings(Path("configs/default.yaml"))
    assert settings.transport.mode == "http"
