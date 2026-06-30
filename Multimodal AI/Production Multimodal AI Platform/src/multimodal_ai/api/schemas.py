"""Pydantic schemas for FastAPI contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from multimodal_ai.domain import InputPayload, RequestEnvelope, TraceContext


class APIRequest(BaseModel):
    """Generic API request payload."""

    input: InputPayload = Field(default_factory=InputPayload)
    options: dict[str, Any] = Field(default_factory=dict)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    trace: TraceContext = Field(default_factory=TraceContext)

    def to_envelope(self) -> RequestEnvelope:
        """Convert API request to domain envelope."""

        return RequestEnvelope(
            input=self.input,
            options=self.options,
            model_overrides=self.model_overrides,
            trace=self.trace,
        )
