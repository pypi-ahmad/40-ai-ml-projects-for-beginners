"""Typed contracts for ingestion, retrieval, and evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Canonical document representation before chunking."""

    doc_id: str
    source: str
    text: str
    title: str | None = None
    filename: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    published_date: str | None = None
    language: str | None = None
    url: str | None = None
    document_hash: str
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Chunk produced from a source document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str
    top_k: int = 10
    mode: Literal["semantic", "lexical", "hybrid"] = "hybrid"
    filters: dict[str, Any] = Field(default_factory=dict)
    rerank: bool = True
    similarity_threshold: float | None = None
    use_cache: bool = True


class SearchHit(BaseModel):
    """Single ranked hit returned by retrieval layer."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    rerank_score: float | None = None
    retrieval_source: Literal["semantic", "lexical", "hybrid", "mmr"] = "semantic"
    rank: int


class SearchResponse(BaseModel):
    """Search results with latency diagnostics."""

    request: SearchRequest
    hits: list[SearchHit]
    latency_ms: float
    vector_latency_ms: float | None = None
    lexical_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationCase(BaseModel):
    """Labeled query used for offline evaluation."""

    query_id: str
    query: str
    category: Literal["technical", "general", "reasoning", "comparison", "multi_document"]
    relevant_doc_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class MetricRow(BaseModel):
    """Evaluation metrics for one retrieval configuration."""

    system_name: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    avg_latency_ms: float


class BenchmarkRun(BaseModel):
    """Benchmark result for model/chunk/retrieval setup."""

    run_id: str
    embedding_model: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    retriever_mode: str
    reranker_enabled: bool
    index_size_bytes: int
    embedding_dim: int
    avg_query_latency_ms: float
    p95_query_latency_ms: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


class SearchLogEvent(BaseModel):
    """Structured log payload for search telemetry."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    query: str
    mode: str
    top_k: int
    latency_ms: float
    hit_count: int
    success: bool
