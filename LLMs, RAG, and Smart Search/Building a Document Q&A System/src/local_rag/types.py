"""Shared typed models for enterprise document Q&A pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


def now_utc() -> datetime:
    """Return timezone-aware UTC timestamp."""

    return datetime.now(UTC)


@dataclass(slots=True)
class LoadedDocument:
    """Normalized document unit loaded from source files."""

    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ChunkRecord:
    """Chunk produced by text splitter."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class RetrievalResult:
    """Single retrieval hit with score and provenance."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    strategy: Literal["vector", "keyword", "hybrid"] = "vector"
    vector_score: float | None = None
    keyword_score: float | None = None


@dataclass(slots=True)
class CitationRecord:
    """Evidence unit attached to final answer."""

    document_name: str
    source_path: str
    page_number: int | None
    chunk_id: str
    similarity_score: float
    evidence_text: str


@dataclass(slots=True)
class TimingBreakdown:
    """Latency values in milliseconds for one RAG query."""

    embedding_ms: float
    retrieval_ms: float
    prompt_ms: float
    generation_ms: float
    total_ms: float


@dataclass(slots=True)
class RAGResponse:
    """End-to-end RAG output bundle."""

    query: str
    answer: str
    model: str
    top_k: int
    retrieval_strategy: Literal["vector", "keyword", "hybrid"]
    citations: list[dict[str, Any]]
    retrieved: list[RetrievalResult]
    timings: TimingBreakdown
    created_at: datetime = field(default_factory=now_utc)


@dataclass(slots=True)
class StreamSession:
    """State captured before token streaming starts."""

    query: str
    model: str
    top_k: int
    retrieval_strategy: Literal["vector", "keyword", "hybrid"]
    retrieved: list[RetrievalResult]
    citations: list[dict[str, Any]]
    embedding_ms: float
    retrieval_ms: float
    prompt_ms: float
    started_total: float


@dataclass(slots=True)
class EvalExample:
    """Ground-truth retrieval/generation evaluation row."""

    query: str
    relevant_doc_ids: list[str]
    relevant_chunk_ids: list[str] = field(default_factory=list)
    answer: str | None = None


@dataclass(slots=True)
class RetrievalMetrics:
    """Aggregate retrieval metrics for one K and strategy."""

    k: int
    strategy: Literal["vector", "keyword", "hybrid"]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    avg_retrieval_latency_ms: float


@dataclass(slots=True)
class GenerationMetrics:
    """Aggregate generation-level metrics."""

    strategy: Literal["vector", "keyword", "hybrid"]
    avg_generation_latency_ms: float
    avg_answer_length: float
    avg_citation_count: float
    bleu: float
    rouge_l: float
    rouge_1: float
    meteor: float
    bertscore_f1: float
    groundedness: float
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float
    hallucination_rate: float


@dataclass(slots=True)
class ResponseMetrics:
    """Backward-compatible response summary metrics."""

    avg_generation_latency_ms: float
    avg_answer_length: float
    avg_citation_count: float


@dataclass(slots=True)
class JudgeScore:
    """LLM-as-a-judge output."""

    correctness: int
    groundedness: int
    completeness: int
    faithfulness: int
    conciseness: int
    citation_quality: int
    rationale: str


@dataclass(slots=True)
class JudgeAggregate:
    """Aggregate LLM-judge metrics over multiple rows."""

    count: int
    correctness: float
    groundedness: float
    completeness: float
    faithfulness: float
    conciseness: float
    citation_quality: float
