from __future__ import annotations

from crew_platform.config import load_agent_catalog, load_settings


def test_settings_load() -> None:
    settings = load_settings("configs/settings.yaml")
    assert settings.llm.default_model
    assert settings.orchestration.max_parallel_tasks == 2


def test_agent_catalog_has_enterprise_team() -> None:
    catalog = load_agent_catalog("configs/agents.yaml")
    roles = {agent.role for agent in catalog.agents}
    assert "Executive Planner" in roles
    assert "Report Generator" in roles
    assert len(catalog.agents) >= 15
