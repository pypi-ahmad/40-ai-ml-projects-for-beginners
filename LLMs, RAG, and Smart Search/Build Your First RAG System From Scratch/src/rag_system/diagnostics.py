"""Diagnostics helpers for retrieval/embedding/index integrity audits."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rag_system.embeddings import EmbeddingEngine
from rag_system.retrieval import RetrievalEngine
from rag_system.types import ChunkRecord, QueryRecord

logger = logging.getLogger(__name__)


def embedding_integrity_report(
    embedding_engine: EmbeddingEngine,
    texts: list[str],
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run sanity checks on embedding outputs and batching behavior."""
    if not texts:
        return {
            "num_texts": 0,
            "dimension": 0,
            "nan_vectors": 0,
            "inf_vectors": 0,
            "duplicate_ratio": 0.0,
            "batch_consistent": True,
            "batch_min_cosine_similarity": 1.0,
        }

    batch_vectors = embedding_engine.embed_batch(texts, batch_size=batch_size)
    single_vectors = [embedding_engine.embed(text) for text in texts[: min(10, len(texts))]]

    arr = np.asarray(batch_vectors, dtype=np.float64)
    dims = int(arr.shape[1]) if arr.ndim == 2 else 0
    nan_vectors = int(np.isnan(arr).any(axis=1).sum()) if arr.ndim == 2 else 0
    inf_vectors = int(np.isinf(arr).any(axis=1).sum()) if arr.ndim == 2 else 0

    # Approximate duplicate ratio by hashing rounded vectors.
    rounded = np.round(arr, 6) if arr.ndim == 2 else np.empty((0, 0))
    hashes = [hash(tuple(row.tolist())) for row in rounded] if rounded.size else []
    duplicate_ratio = float(1.0 - (len(set(hashes)) / len(hashes))) if hashes else 0.0

    min_cosine_similarity = 1.0
    for idx, single in enumerate(single_vectors):
        if idx >= len(batch_vectors):
            break
        single_arr = np.asarray(single, dtype=np.float64)
        batch_arr = np.asarray(batch_vectors[idx], dtype=np.float64)
        denom = np.linalg.norm(single_arr) * np.linalg.norm(batch_arr)
        cosine = 0.0 if denom == 0.0 else float(np.dot(single_arr, batch_arr) / denom)
        min_cosine_similarity = min(min_cosine_similarity, cosine)
    batch_consistent = min_cosine_similarity >= 0.995

    return {
        "num_texts": len(texts),
        "dimension": dims,
        "nan_vectors": nan_vectors,
        "inf_vectors": inf_vectors,
        "duplicate_ratio": duplicate_ratio,
        "batch_consistent": batch_consistent,
        "batch_min_cosine_similarity": float(min_cosine_similarity),
    }


def index_integrity_report(
    retrieval_engine: RetrievalEngine,
    expected_chunks: list[ChunkRecord],
) -> dict[str, Any]:
    """Validate Chroma collection count/id consistency against source chunks."""
    expected_ids = {chunk.chunk_id for chunk in expected_chunks}
    count_in_collection = retrieval_engine.collection.count()

    sample_size = min(count_in_collection, 5000)
    snapshot = retrieval_engine.collection.get(include=["metadatas"], limit=sample_size)
    ids = snapshot.get("ids", []) or []
    metadatas = snapshot.get("metadatas", []) or []

    missing_doc_meta = 0
    for meta in metadatas:
        if not meta or "doc_id" not in meta:
            missing_doc_meta += 1

    overlap = len(set(ids) & expected_ids)

    return {
        "collection_count": count_in_collection,
        "expected_chunk_count": len(expected_chunks),
        "sample_size": sample_size,
        "sample_overlap_with_expected_ids": overlap,
        "sample_missing_doc_metadata": missing_doc_meta,
        "count_matches_expected": bool(count_in_collection == len(expected_chunks)),
    }


def retrieval_diagnostics(
    retrieval_engine: RetrievalEngine,
    queries: list[QueryRecord],
    top_k: int,
    min_relevance_score: float,
    max_queries: int = 300,
) -> list[dict[str, Any]]:
    """Collect per-query retrieval diagnostics with failure buckets."""
    rows: list[dict[str, Any]] = []
    for query in queries[:max_queries]:
        hits = retrieval_engine.query(
            query=query.query,
            top_k=top_k,
            dedupe_by_doc=True,
        )
        bucket = retrieval_engine.classify_retrieval(
            chunks=hits,
            gold_doc_ids=query.gold_doc_ids,
            min_relevance_score=min_relevance_score,
        )
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "top_score": hits[0].score if hits else 0.0,
                "num_hits": len(hits),
                "gold_doc_ids": query.gold_doc_ids,
                "retrieved_doc_ids": [hit.doc_id for hit in hits],
                "failure_bucket": bucket,
            }
        )
    return rows
