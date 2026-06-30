"""Domain models and API contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceContext(BaseModel):
    """Trace context propagated across calls."""

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str = "unknown"


class InputPayload(BaseModel):
    """Incoming request payload."""

    text: str | None = None
    question: str | None = None
    image_path: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    document_path: str | None = None
    query: str | None = None


class RequestEnvelope(BaseModel):
    """Normalized request envelope for all interfaces."""

    input: InputPayload = Field(default_factory=InputPayload)
    options: dict[str, Any] = Field(default_factory=dict)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    trace: TraceContext = Field(default_factory=TraceContext)


class ErrorInfo(BaseModel):
    """Structured error information."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ResponseEnvelope(BaseModel):
    """Normalized response envelope for all outputs."""

    status: Literal["ok", "error"]
    result: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    latency_ms: float = 0.0
    artifacts: dict[str, Any] = Field(default_factory=dict)
    trace_id: str
    errors: list[ErrorInfo] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AssetRecord(BaseModel):
    """Metadata record for asset."""

    asset_id: str = Field(default_factory=lambda: str(uuid4()))
    path: Path
    media_type: str
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)


class OCRBlock(BaseModel):
    """OCR token or line with coordinates."""

    text: str
    bbox: list[float] = Field(default_factory=list)
    confidence: float | None = None


class OCRResult(BaseModel):
    """OCR output container."""

    engine: str
    text: str
    blocks: list[OCRBlock] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    is_scanned: bool = False


class RetrievalHit(BaseModel):
    """Vector retrieval hit."""

    id: str
    score: float
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
