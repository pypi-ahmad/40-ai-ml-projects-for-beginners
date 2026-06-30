"""FastAPI request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from langgraph_platform.state.models import HITLAction


class ChatRequest(BaseModel):
    """Chat/workflow request."""

    message: str = Field(..., min_length=1)
    session_id: str | None = None


class WorkflowRequest(BaseModel):
    """Explicit workflow execution request."""

    user_request: str = Field(..., min_length=1)
    session_id: str | None = None


class HITLRequest(BaseModel):
    """Human-in-the-loop action request."""

    workflow_id: str
    action: HITLAction
    note: str | None = None


class SearchRequest(BaseModel):
    """Search endpoint payload."""

    query: str
    max_results: int = 5


class KnowledgeIngestRequest(BaseModel):
    """Knowledge ingestion payload."""

    paths: list[str] = []
    urls: list[str] = []


class ReportExportRequest(BaseModel):
    """Report export request."""

    workflow_id: str
    markdown_report: str
    payload: dict[str, Any] = Field(default_factory=dict)
