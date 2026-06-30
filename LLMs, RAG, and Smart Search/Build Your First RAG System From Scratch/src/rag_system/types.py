"""Core data types for the educational RAG system.

This module centralizes strongly typed records used across ingestion,
chunking, retrieval, generation, and evaluation stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentRecord:
    """Canonical document record stored before chunking.

    Attributes:
        doc_id: Stable document identifier.
        text: Full document text content.
        metadata: Additional document metadata (title, split, source, ...).
    """

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    """Single chunk record stored in vector database.

    Attributes:
        chunk_id: Stable chunk identifier.
        doc_id: Parent document id.
        text: Chunk text used for retrieval and context building.
        metadata: Chunk metadata (strategy, indices, token estimates, ...).
        parent_id: Optional parent chunk id for parent-child chunking.
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None


@dataclass(slots=True)
class QueryRecord:
    """Evaluation query with optional gold references.

    Attributes:
        query_id: Stable query identifier.
        query: User question.
        gold_doc_ids: Relevant document ids for retrieval metrics.
        gold_answer: Reference answer text for generation metrics.
        metadata: Extra metadata (answerable, title, split, ...).
    """

    query_id: str
    query: str
    gold_doc_ids: list[str] = field(default_factory=list)
    gold_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """Retrieved chunk returned by retriever.

    Attributes:
        chunk_id: Chunk identifier.
        doc_id: Parent document id.
        text: Retrieved chunk text.
        score: Similarity score normalized to [0, 1] (higher is better).
        distance: Raw Chroma distance (lower is better).
        metadata: Stored chunk metadata.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RAGResponse:
    """Full response payload returned by RAG pipeline."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    context: str
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""


@dataclass(slots=True)
class JudgeResult:
    """LLM-as-a-judge result for one generated answer."""

    relevance: float
    correctness: float
    groundedness: float
    completeness: float
    faithfulness: float
    rationale: str
    raw_output: str
