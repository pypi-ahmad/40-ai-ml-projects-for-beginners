"""Typed response models used across task modules and UI handlers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    """Structured error payload returned to UI and tests."""

    message: str
    stage: str


class SentimentResult(BaseModel):
    """Sentiment classification response."""

    label: Literal["Positive", "Negative", "Neutral", "Error"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""
    model: str
    latency_ms: float = 0.0
    error: ErrorInfo | None = None


class SummaryResult(BaseModel):
    """Summarization response."""

    summary: str
    key_points: list[str]
    model: str
    original_word_count: int
    latency_ms: float = 0.0
    error: ErrorInfo | None = None


class TranslationResult(BaseModel):
    """Translation response."""

    translated_text: str
    source_lang: str
    target_lang: str
    model: str
    latency_ms: float = 0.0
    error: ErrorInfo | None = None


class ChatTurn(BaseModel):
    """Single chat message in conversation history."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatResult(BaseModel):
    """Chat generation response."""

    response: str
    model: str
    turns_kept: int
    latency_ms: float = 0.0
    error: ErrorInfo | None = None


class DocumentResult(BaseModel):
    """Document extraction + QA response."""

    extracted_text: str
    summary: str
    answer: str
    pages_processed: int
    ocr_model_used: str
    qa_model: str
    latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None


class BenchmarkResult(BaseModel):
    """Single model benchmark summary."""

    model: str
    prompt_key: str
    runs: int
    successful_runs: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_tokens_per_sec: float
    mean_memory_mb: float
    cold_start_latency_ms: float = 0.0
    warm_start_latency_ms: float = 0.0
    quality_score: float
    error: str | None = None


class WarmupResult(BaseModel):
    """Model warmup operation status."""

    model: str
    ready: bool
    latency_ms: float
    message: str
