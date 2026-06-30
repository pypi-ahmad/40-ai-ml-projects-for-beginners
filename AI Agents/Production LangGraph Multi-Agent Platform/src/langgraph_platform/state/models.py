"""Typed state contracts for LangGraph workflow runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    """Lifecycle state for workflow nodes."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class VerificationStatus(str, Enum):
    """Verification status of final or intermediate content."""

    UNKNOWN = "unknown"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class HITLAction(str, Enum):
    """Human in the loop actions."""

    APPROVE = "approve"
    REJECT = "reject"
    PAUSE = "pause"
    RESUME = "resume"
    OVERRIDE = "override"
    RERUN = "rerun"


class RoutingDecision(BaseModel):
    """Dynamic routing flags determined by planner/router."""

    require_web_search: bool = True
    require_rag: bool = True
    require_memory: bool = True
    require_code_execution: bool = False
    require_verification: bool = True


class ToolCallRecord(BaseModel):
    """Record of tool invocation."""

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None
    output_preview: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Citation(BaseModel):
    """Citation object for report provenance."""

    source_id: str
    title: str
    url: str | None = None
    snippet: str = ""
    confidence: float = 0.5


class RetrievedDocument(BaseModel):
    """Knowledge artifact used by workflow."""

    doc_id: str
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class AgentOutput(BaseModel):
    """Output from a single agent execution."""

    agent_name: str
    content: str
    structured: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    retries: int = 0


class ExecutionMetadata(BaseModel):
    """Execution metadata and observability fields."""

    workflow_id: str
    session_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    node_status: dict[str, NodeStatus] = Field(default_factory=dict)
    node_durations_ms: dict[str, float] = Field(default_factory=dict)
    retries: dict[str, int] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    active_node: str | None = None


class TokenUsage(BaseModel):
    """Token accounting per model/provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    by_model: dict[str, int] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Strongly typed shared state for graph execution."""

    user_request: str
    execution_plan: str = ""
    subtasks: list[str] = Field(default_factory=list)
    routing: RoutingDecision = Field(default_factory=RoutingDecision)
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    search_results: list[dict[str, Any]] = Field(default_factory=list)
    memory: list[dict[str, Any]] = Field(default_factory=list)
    intermediate_outputs: dict[str, AgentOutput] = Field(default_factory=dict)
    reports: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN
    confidence_score: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    execution_metadata: ExecutionMetadata
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    hitl_actions: list[HITLAction] = Field(default_factory=list)
    paused: bool = False


class WorkflowResult(BaseModel):
    """Final materialized result from workflow."""

    workflow_id: str
    final_report: str
    confidence: float
    verification_status: VerificationStatus
    citations: list[Citation]
    metadata: ExecutionMetadata
