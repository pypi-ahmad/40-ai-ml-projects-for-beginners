from __future__ import annotations

from pathlib import Path

import yaml

from reasoning_agent.config import clear_settings_cache, get_settings


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_env_override_precedence(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_yaml(config_path, {"llm": {"base_url": "http://yaml.local:11434"}})

    clear_settings_cache()
    monkeypatch.setenv("AGENT__LLM__BASE_URL", "http://env.local:11434")
    settings = get_settings(config_path=config_path, refresh=True)
    assert settings.llm.base_url == "http://env.local:11434"

    monkeypatch.delenv("AGENT__LLM__BASE_URL")
    settings = get_settings(config_path=config_path, refresh=True)
    assert settings.llm.base_url == "http://yaml.local:11434"


def test_deprecated_use_langgraph_runtime_key_maps_to_runtime_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy.yaml"
    _write_yaml(config_path, {"agent": {"use_langgraph_runtime": False}})

    clear_settings_cache()
    settings = get_settings(config_path=config_path, refresh=True)
    assert settings.agent.runtime_mode == "fallback"

    _write_yaml(config_path, {"agent": {"use_langgraph_runtime": True}})
    settings = get_settings(config_path=config_path, refresh=True)
    assert settings.agent.runtime_mode == "graph"
