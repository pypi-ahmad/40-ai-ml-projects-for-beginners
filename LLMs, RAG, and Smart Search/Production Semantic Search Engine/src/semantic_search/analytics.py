"""Analytics and observability helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from semantic_search.schemas import DocumentChunk, DocumentRecord, SearchLogEvent


def compute_collection_analytics(
    documents: list[DocumentRecord],
    chunks: list[DocumentChunk],
    embedding_model: str,
    vector_db_size_bytes: int,
) -> dict[str, Any]:
    """Compute corpus and index analytics summary."""
    categories = Counter([doc.category or "unknown" for doc in documents])
    languages = Counter([doc.language or "unknown" for doc in documents])

    chunk_lengths = [len(chunk.text.split()) for chunk in chunks]
    return {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "embedding_model": embedding_model,
        "vector_db_size_bytes": vector_db_size_bytes,
        "average_chunk_length": mean(chunk_lengths) if chunk_lengths else 0.0,
        "category_distribution": dict(categories),
        "language_distribution": dict(languages),
    }


def write_search_log(path: str | Path, event: SearchLogEvent) -> None:
    """Append search telemetry event as JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(event.model_dump_json() + "\n")


def summarize_search_logs(path: str | Path) -> dict[str, Any]:
    """Aggregate search telemetry for dashboard."""
    target = Path(path)
    if not target.exists():
        return {
            "total_searches": 0,
            "success_rate": 0.0,
            "avg_latency_ms": 0.0,
            "top_terms": {},
        }

    events: list[dict[str, Any]] = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]
    if not events:
        return {
            "total_searches": 0,
            "success_rate": 0.0,
            "avg_latency_ms": 0.0,
            "top_terms": {},
        }

    success_count = sum(1 for event in events if event.get("success"))
    avg_latency = mean([float(event.get("latency_ms", 0.0)) for event in events])

    term_counter: Counter[str] = Counter()
    for event in events:
        for token in str(event.get("query", "")).split():
            term_counter[token.lower()] += 1

    return {
        "total_searches": len(events),
        "success_rate": success_count / len(events),
        "avg_latency_ms": avg_latency,
        "top_terms": dict(term_counter.most_common(20)),
    }
