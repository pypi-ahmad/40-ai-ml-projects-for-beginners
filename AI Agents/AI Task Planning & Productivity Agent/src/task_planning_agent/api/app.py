"""FastAPI app factory."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from task_planning_agent.api.routers.core import router
from task_planning_agent.config import load_config
from task_planning_agent.logging_utils import configure_logging


def create_app() -> FastAPI:
    """Create FastAPI app instance."""

    configure_logging()
    app = FastAPI(
        title="AI Task Planning & Productivity Agent",
        version="0.1.0",
        description="Production-grade planning assistant with LangGraph + local LLM support",
    )
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    cfg = load_config()
    api_cfg = cfg.api
    uvicorn.run(
        "task_planning_agent.api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=int(api_cfg.get("port", 8000)),
        reload=False,
    )


if __name__ == "__main__":
    run()
