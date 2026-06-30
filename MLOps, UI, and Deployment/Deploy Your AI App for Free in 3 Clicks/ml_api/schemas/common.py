"""Common schema models used across API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorPayload(BaseModel):
    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: dict[str, str | int | float] = Field(default_factory=dict)
    request_id: str = Field(description="Request identifier for debugging")


class ErrorResponse(BaseModel):
    error: ErrorPayload
