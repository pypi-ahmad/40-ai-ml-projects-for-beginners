"""FastAPI request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model_id: str | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    model_id: str | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class BatchRequest(BaseModel):
    prompts: list[str]
    model_id: str | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    tokens_generated: int


class HealthResponse(BaseModel):
    status: str
    version: str
