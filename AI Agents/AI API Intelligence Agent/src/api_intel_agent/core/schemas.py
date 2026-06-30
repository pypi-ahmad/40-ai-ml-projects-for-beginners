"""Core schema contracts shared across API, graph, and tools."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"


class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    RUNNING = "running"


class SourceRecord(BaseModel):
    provider: str
    endpoint: str
    url: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChartArtifact(BaseModel):
    title: str
    kind: str
    path: str


class ErrorRecord(BaseModel):
    code: str
    message: str
    provider: str | None = None
    retryable: bool = False


class ConnectorResult(BaseModel):
    provider: str
    endpoint: str
    status: str
    records: list[dict[str, Any]] = Field(default_factory=list)
    pagination: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    cached: bool = False
    error: ErrorRecord | None = None


class AnalyzeRequest(BaseModel):
    query: str
    model: str | None = None
    apis: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    output_format: OutputFormat = OutputFormat.MARKDOWN
    use_memory: bool = True
    use_cache: bool = True
    max_parallel_calls: int = 4


class AnalyzeResponse(BaseModel):
    run_id: str
    status: RunStatus
    summary: str
    insights: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    charts: list[ChartArtifact] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    errors: list[ErrorRecord] = Field(default_factory=list)


class QueryHistoryItem(BaseModel):
    run_id: str
    query: str
    status: RunStatus
    created_at: datetime


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 5


class MemorySearchHit(BaseModel):
    id: str
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
