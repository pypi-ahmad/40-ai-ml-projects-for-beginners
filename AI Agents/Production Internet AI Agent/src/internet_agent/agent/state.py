"""State schema for multi-agent LangGraph workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    session_id: str
    user_query: str
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    intent: str = ""
    plan_steps: list[str] = Field(default_factory=list)
    planned_tools: list[dict[str, Any]] = Field(default_factory=list)

    need_internet: bool = False
    selected_providers: list[str] = Field(default_factory=list)

    search_results: list[dict[str, Any]] = Field(default_factory=list)
    semantic_hits: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_documents: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)

    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)

    draft_answer: str = ""
    final_answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)

    confidence: float = 0.0
    hallucination_risk: str = "unknown"
    missing_info: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    retry_search: bool = False

    verification_loops: int = 0
    done: bool = False
    error: str = ""

    report_payload: dict[str, Any] = Field(default_factory=dict)

    def can_retry_search(self, max_loops: int) -> bool:
        return self.retry_search and self.verification_loops < max_loops
