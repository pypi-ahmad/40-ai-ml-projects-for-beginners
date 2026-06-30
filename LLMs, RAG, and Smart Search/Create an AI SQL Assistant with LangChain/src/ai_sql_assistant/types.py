"""Typed interfaces used across assistant pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Natural-language question request."""

    question: str
    user_id: str = "default"
    persona: str = "Business Analyst"
    conversation_id: str = "default"


class GenerationCandidate(BaseModel):
    """Generated SQL candidate with metadata."""

    sql: str
    approach: Literal["langchain", "direct"]
    model: str
    prompt_name: str
    latency_ms: float
    fallback_used: bool = False
    raw_response: str | None = None


class ValidationIssue(BaseModel):
    """Validation finding for generated SQL."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


class ValidationReport(BaseModel):
    """Final validation result."""

    is_valid: bool
    normalized_sql: str
    issues: list[ValidationIssue] = Field(default_factory=list)
    blocked_keywords: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Query execution output and metrics."""

    status: Literal["success", "error", "blocked"]
    sql: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error_message: str | None = None
    explain_plan: list[dict[str, Any]] = Field(default_factory=list)
    complexity_score: float = 0.0


class VisualizationSpec(BaseModel):
    """Chart recommendation and fields."""

    chart_type: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    reason: str = ""


class ConversationTurn(BaseModel):
    """Persistent conversation turn."""

    conversation_id: str
    user_id: str
    question: str
    sql: str
    explanation: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkCase(BaseModel):
    """Single benchmark entry."""

    case_id: str
    category: str
    question: str
    ground_truth_sql: str
    tags: list[str] = Field(default_factory=list)


class BenchmarkCaseResult(BaseModel):
    """Per-case output for one model/approach."""

    case_id: str
    model: str
    approach: str
    generated_sql: str
    exact_match: bool
    execution_accuracy: bool
    result_correctness: bool
    generation_latency_ms: float
    execution_latency_ms: float
    complexity_score: float
    row_count: int = 0
    token_count_estimate: int = 0
    memory_mb: float = 0.0
    error: str | None = None


class JudgeScore(BaseModel):
    """Judge scoring rubric."""

    sql_correctness: float = Field(ge=0.0, le=1.0)
    business_correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    readability: float = Field(ge=0.0, le=1.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    rationale: str


class BenchmarkRun(BaseModel):
    """Aggregate benchmark result."""

    run_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    results: list[BenchmarkCaseResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class AssistantResponse(BaseModel):
    """Top-level response returned by assistant pipeline."""

    request: QueryRequest
    generation: GenerationCandidate
    validation: ValidationReport
    execution: ExecutionResult
    explanation: str
    business_interpretation: str
    visualization_options: list[VisualizationSpec] = Field(default_factory=list)
