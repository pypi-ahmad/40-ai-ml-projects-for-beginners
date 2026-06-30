"""Run one demo query and print structured output."""

from __future__ import annotations

import asyncio
import os

from reasoning_agent.agent.models import AgentRequest
from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.config import get_settings


async def main() -> None:
    settings = get_settings()
    settings.memory.chroma_enabled = False
    settings.agent.runtime_mode = "fallback"
    settings.agent.use_llm_for_planning = False
    settings.agent.use_llm_for_response = False
    os.environ.setdefault("AGENT_OFFLINE_MODE", "1")

    runner = AgentRunner(settings)
    result = await runner.run(
        AgentRequest(
            query="Convert 10 km to miles and explain method",
            session_id="demo",
        )
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
