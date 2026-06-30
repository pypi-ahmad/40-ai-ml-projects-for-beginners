"""Event models for tracing and analytics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class EventRecord(BaseModel):
    """Structured event record."""

    event_type: str
    run_id: str
    status: Literal["ok", "error", "warn", "info"] = "info"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """Tool call specific telemetry record."""

    run_id: str
    tool_name: str
    latency_ms: float
    success: bool
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
