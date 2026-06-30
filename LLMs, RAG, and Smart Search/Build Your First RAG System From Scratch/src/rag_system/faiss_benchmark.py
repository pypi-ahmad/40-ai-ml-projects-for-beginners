"""Optional FAISS benchmark utilities for appendix comparison."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from rag_system.embeddings import EmbeddingEngine
from rag_system.types import ChunkRecord

logger = logging.getLogger(__name__)


class FaissUnavailableError(RuntimeError):
    """Raised when FAISS optional dependency is missing."""


class FaissBenchmark:
    """Simple FAISS retrieval benchmark for ChromaDB appendix comparison."""

    def __init__(self, embedding_engine: EmbeddingEngine | None = None) -> None:
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.index = None
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.doc_ids: list[str] = []

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        """Build cosine-normalized FAISS index from chunk embeddings."""
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise FaissUnavailableError(
                "FAISS is not installed. Run: uv sync --extra faiss"
            ) from exc

        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunk_texts = [chunk.text for chunk in chunks]
        self.doc_ids = [chunk.doc_id for chunk in chunks]

        vectors = np.asarray(self.embedding_engine.embed_batch(self.chunk_texts, batch_size=32), dtype="float32")
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index
        logger.info("Built FAISS index with %d vectors", vectors.shape[0])

    def query(self, query_text: str, top_k: int = 6) -> list[dict[str, Any]]:
        """Query FAISS index and return ranked chunk metadata."""
        if self.index is None:
            raise RuntimeError("FAISS index not built")

        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise FaissUnavailableError(
                "FAISS is not installed. Run: uv sync --extra faiss"
            ) from exc

        query_vec = np.asarray([self.embedding_engine.embed(query_text)], dtype="float32")
        faiss.normalize_L2(query_vec)

        start = time.perf_counter()
        scores, indices = self.index.search(query_vec, top_k)
        latency = time.perf_counter() - start

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append(
                {
                    "rank": rank,
                    "chunk_id": self.chunk_ids[idx],
                    "doc_id": self.doc_ids[idx],
                    "score": float(score),
                    "text": self.chunk_texts[idx],
                    "latency_s": latency,
                }
            )
        return results
