"""Pydantic schemas for API contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    confidence: float
    hallucination_risk: str
    citations: list[dict[str, Any]]
    reasoning_trace: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]]


class SearchRequest(BaseModel):
    session_id: str = Field(default="default")
    query: str
    providers: list[str] | None = None


class SearchResponse(BaseModel):
    query: str
    providers: list[str]
    results: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    from_cache: bool
    latency_ms: float


class BrowseRequest(BaseModel):
    session_id: str = Field(default="default")
    url: str


class ReportRequest(BaseModel):
    session_id: str = Field(default="default")
    format: str = Field(default="json", description="markdown|html|pdf|json")
    payload: dict[str, Any]


class MemoryRequest(BaseModel):
    query: str
    top_k: int | None = None
