"""Retriever abstraction for local RAG."""

from __future__ import annotations

import time
from typing import Any

from local_rag.embeddings import OllamaEmbeddingClient
from local_rag.types import RetrievalResult
from local_rag.vectordb import ChromaVectorStore


class Retriever:
    """Top-K retriever with metadata filtering support."""

    def __init__(self, vector_store: ChromaVectorStore, embedder: OllamaEmbeddingClient) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievalResult], float]:
        """Retrieve top-k chunks and retrieval latency in ms."""

        query_embedding = self.embedder.embed_texts([query])[0]

        started = time.perf_counter()
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=filters,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        return results, retrieval_ms
