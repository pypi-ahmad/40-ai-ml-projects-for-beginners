from __future__ import annotations

import pytest

from crew_platform.config import load_agent_catalog, load_settings
from crew_platform.orchestration.catalog import AgentCatalog
from crew_platform.orchestration.planner import PlannerService


class FailingLLM:
    async def generate(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("offline")


@pytest.mark.asyncio
async def test_planner_fallback_generates_dag() -> None:
    settings = load_settings("configs/settings.yaml")
    catalog = AgentCatalog(load_agent_catalog("configs/agents.yaml"))
    planner = PlannerService(settings=settings, llm=FailingLLM(), catalog=catalog)

    plan = await planner.build_plan("Build enterprise SEO and finance strategy")
    task_ids = [task.task_id for task in plan.tasks]

    assert plan.run_id.startswith("run-")
    assert "plan_scope" in task_ids
    assert "final_report" in task_ids
    assert any(task.dependencies for task in plan.tasks if task.task_id != "plan_scope")
