"""Runtime service layer used by Streamlit pages."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from reasoning_agent.agent.models import AgentRequest
from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.config import Settings, get_settings


@lru_cache(maxsize=1)
def _runner_cached(config_path: str | None = None) -> AgentRunner:
    settings: Settings = get_settings(config_path=config_path)
    return AgentRunner(settings=settings)


def run_agent_query(query: str, session_id: str = "streamlit") -> dict[str, Any]:
    runner = _runner_cached()
    result = asyncio.run(runner.run(AgentRequest(query=query, session_id=session_id)))
    return result.model_dump(mode="json")
