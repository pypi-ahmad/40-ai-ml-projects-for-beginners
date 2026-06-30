"""Core orchestration models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class TaskSpec(BaseModel):
    task_id: str
    title: str
    description: str
    agent_role: str
    dependencies: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    output_schema: str = "generic"
    retries_allowed: int = 2


class TaskExecution(TaskSpec):
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    confidence: float = 0.0


class PlanProposal(BaseModel):
    run_id: str
    objective: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tasks: list[TaskSpec]
    assumptions: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    factual_score: float = 0.0
    qa_score: float = 0.0
    reflection_score: float = 0.0
    confidence: float = 0.0
    issues: list[str] = Field(default_factory=list)
    needs_rerun: bool = False


class ReportArtifact(BaseModel):
    run_id: str
    title: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str
    sections: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrewRunRequest(BaseModel):
    query: str
    session_id: str = "default"
    auto_execute: bool = False
    force_consensus: bool = False


class PlanApproval(BaseModel):
    approved: bool
    reviewer: str = "human"
    feedback: str | None = None


class CrewRunResult(BaseModel):
    run_id: str
    status: RunStatus
    plan: PlanProposal
    tasks: list[TaskExecution] = Field(default_factory=list)
    verification: VerificationResult | None = None
    report: ReportArtifact | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
