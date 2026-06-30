from __future__ import annotations

from internet_agent.config import Settings, get_settings


def test_settings_load() -> None:
    settings = get_settings("configs/config.yaml")
    assert isinstance(settings, Settings)
    assert settings.llm.planning_model
    assert "qwen3" in "".join(settings.models.supported_families)
