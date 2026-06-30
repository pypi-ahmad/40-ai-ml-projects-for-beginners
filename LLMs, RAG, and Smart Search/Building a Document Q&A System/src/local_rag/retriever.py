"""Retriever abstraction with vector, keyword, and hybrid strategies."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Literal

from local_rag.embeddings import OllamaEmbeddingClient
from local_rag.lexical import BM25LexicalIndex
from local_rag.types import RetrievalResult
from local_rag.vectordb import ChromaVectorStore

RetrievalStrategy = Literal["vector", "keyword", "hybrid"]


class Retriever:
    """Top-K retriever with metadata filtering and hybrid support."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: OllamaEmbeddingClient,
        lexical_index: BM25LexicalIndex,
        *,
        rrf_k: int = 60,
        vector_weight: float = 0.55,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.lexical_index = lexical_index
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        strategy: RetrievalStrategy = "vector",
    ) -> tuple[list[RetrievalResult], float]:
        """Retrieve top-k chunks by selected strategy."""

        started = time.perf_counter()

        if strategy == "keyword":
            results = self.lexical_index.query(query, top_k=top_k, filters=filters)
            retrieval_ms = (time.perf_counter() - started) * 1000
            return results, retrieval_ms

        if strategy == "vector":
            query_embedding = self.embedder.embed_texts([query])[0]
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=top_k,
                where=filters,
            )
            retrieval_ms = (time.perf_counter() - started) * 1000
            return results, retrieval_ms

        # hybrid: union vector + keyword, rank with weighted reciprocal rank fusion.
        vector_k = max(top_k * 2, top_k)
        keyword_k = max(top_k * 2, top_k)

        query_embedding = self.embedder.embed_texts([query])[0]
        vector_hits = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=vector_k,
            where=filters,
        )
        keyword_hits = self.lexical_index.query(
            query,
            top_k=keyword_k,
            filters=filters,
        )

        ranked = self._hybrid_rrf(vector_hits=vector_hits, keyword_hits=keyword_hits, top_k=top_k)
        retrieval_ms = (time.perf_counter() - started) * 1000
        return ranked, retrieval_ms

    def _hybrid_rrf(
        self,
        *,
        vector_hits: list[RetrievalResult],
        keyword_hits: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        vector_ranks = {row.chunk_id: idx + 1 for idx, row in enumerate(vector_hits)}
        keyword_ranks = {row.chunk_id: idx + 1 for idx, row in enumerate(keyword_hits)}

        by_id: dict[str, RetrievalResult] = {}
        for row in vector_hits:
            by_id[row.chunk_id] = replace(
                row,
                strategy="hybrid",
                vector_score=row.score,
                keyword_score=0.0,
            )
        for row in keyword_hits:
            if row.chunk_id in by_id:
                existing = by_id[row.chunk_id]
                existing.keyword_score = row.score
            else:
                by_id[row.chunk_id] = replace(
                    row,
                    strategy="hybrid",
                    vector_score=0.0,
                    keyword_score=row.score,
                )

        fused_rows: list[tuple[float, RetrievalResult]] = []
        for chunk_id, row in by_id.items():
            vector_rank = vector_ranks.get(chunk_id)
            keyword_rank = keyword_ranks.get(chunk_id)

            vector_term = 0.0
            if vector_rank is not None:
                vector_term = self.vector_weight * (1.0 / (self.rrf_k + vector_rank))

            keyword_term = 0.0
            if keyword_rank is not None:
                keyword_weight = 1.0 - self.vector_weight
                keyword_term = keyword_weight * (1.0 / (self.rrf_k + keyword_rank))

            fused = vector_term + keyword_term
            row.score = fused
            fused_rows.append((fused, row))

        fused_rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in fused_rows[:top_k]]
