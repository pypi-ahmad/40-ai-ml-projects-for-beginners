"""Application entry helpers."""

from __future__ import annotations

from langgraph_platform.api.main import create_app
from langgraph_platform.config.loader import load_config
from langgraph_platform.engine.workflow import LangGraphWorkflowEngine


def build_engine() -> LangGraphWorkflowEngine:
    """Create configured workflow engine instance."""

    return LangGraphWorkflowEngine(load_config())


def build_api_app():
    """Create configured FastAPI app."""

    return create_app()
