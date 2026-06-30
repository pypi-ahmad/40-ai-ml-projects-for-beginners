"""Typed schemas for ingestion, retrieval, generation, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def now_utc() -> datetime:
    """Return timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class RetrievalMode(str, Enum):
    """Retrieval routes supported by the orchestrator."""

    AUTO = "auto"
    LOCAL = "local"
    WEB = "web"
    HYBRID = "hybrid"


class ChunkingStrategy(str, Enum):
    """Chunking strategy options."""

    RECURSIVE = "recursive"
    TOKEN = "token"
    SEMANTIC = "semantic"


@dataclass(slots=True)
class DocumentRecord:
    """Normalized document payload loaded from source files."""

    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ChunkRecord:
    """Chunk payload ready for embedding and indexing."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class RetrievedContext:
    """Retrieved context item after retrieval or reranking."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    source: str


@dataclass(slots=True)
class Citation:
    """Citation payload shown to users for each answer."""

    source_file: str
    page_number: int | None
    url: str | None
    chunk_id: str
    confidence: float
    title: str | None = None


@dataclass(slots=True)
class RouteDecision:
    """Intent and routing decision output."""

    mode: RetrievalMode
    reason: str
    confidence: float


@dataclass(slots=True)
class TimingBreakdown:
    """Latency measurements in milliseconds."""

    intent_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    context_ms: float = 0.0
    generation_ms: float = 0.0
    judge_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class AssistantResponse:
    """Final assistant response payload."""

    query: str
    answer: str
    mode: RetrievalMode
    route_reason: str
    citations: list[Citation]
    retrieved: list[RetrievedContext]
    timings: TimingBreakdown
    prompt_name: str
    judge: dict[str, Any]
    created_at: datetime = field(default_factory=now_utc)


@dataclass(slots=True)
class ConversationTurn:
    """One conversation turn for short-term memory."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=now_utc)


@dataclass(slots=True)
class GraphState:
    """LangGraph state object."""

    query: str
    requested_mode: RetrievalMode
    prompt_name: str
    route: RouteDecision | None = None
    local_context: list[RetrievedContext] = field(default_factory=list)
    web_context: list[RetrievedContext] = field(default_factory=list)
    merged_context: list[RetrievedContext] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    answer: str = ""
    judge: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timings: TimingBreakdown = field(default_factory=TimingBreakdown)


class EvalSample(BaseModel):
    """Evaluation dataset row."""

    id: str
    category: str
    question: str
    expected_mode: RetrievalMode
    expected_keywords: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    reference_answer: str | None = None


class EvalResult(BaseModel):
    """Evaluation result row."""

    sample_id: str
    question: str
    category: str
    mode_used: RetrievalMode
    accuracy: float
    faithfulness: float
    relevance: float
    context_precision: float
    latency_ms: float
    citations_count: int


class EvalReport(BaseModel):
    """Aggregate evaluation report."""

    total_samples: int
    accuracy: float
    faithfulness: float
    relevance: float
    context_precision: float
    mean_latency_ms: float
    precision_at_k: float
    recall_at_k: float
