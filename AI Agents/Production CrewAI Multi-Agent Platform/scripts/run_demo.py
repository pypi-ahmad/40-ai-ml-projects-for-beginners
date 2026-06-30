"""Run demo workflow end-to-end."""

from __future__ import annotations

import asyncio

from loguru import logger

from crew_platform.config import load_settings
from crew_platform.orchestration import CrewRunRequest, PlanApproval, create_service


async def main() -> None:
    service = create_service(load_settings())
    planned = await service.create_plan(
        CrewRunRequest(
            query="Create enterprise launch brief for AI multi-agent collaboration platform",
            session_id="demo",
            auto_execute=False,
        )
    )
    await service.apply_approval(planned.run_id, PlanApproval(approved=True, reviewer="demo"))
    result = await service.execute_run(planned.run_id)
    logger.info("Demo result: {}", result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
