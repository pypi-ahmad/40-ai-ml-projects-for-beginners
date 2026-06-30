"""Semantic and response cache utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

from hybrid_research_assistant.embeddings import EmbeddingProvider
from hybrid_research_assistant.utils import cosine_similarity


@dataclass(slots=True)
class CacheEntry:
    query: str
    response: str
    embedding: list[float]
    mode: str
    expires_at: float


class SemanticResponseCache:
    """In-memory semantic cache for repeated queries."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        similarity_threshold: float,
        ttl_local: int,
        ttl_web: int,
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_local = ttl_local
        self.ttl_web = ttl_web
        self.entries: list[CacheEntry] = []

    def get(self, query: str) -> CacheEntry | None:
        now = time.time()
        self.entries = [entry for entry in self.entries if entry.expires_at > now]
        if not self.entries:
            return None

        query_embedding = self.embedder.embed_texts([query])[0]
        best_entry: CacheEntry | None = None
        best_score = -1.0

        for entry in self.entries:
            score = cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self.similarity_threshold:
            return None
        return best_entry

    def put(self, query: str, response: str, mode: str) -> None:
        ttl = self.ttl_web if mode in {"web", "hybrid"} else self.ttl_local
        embedding = self.embedder.embed_texts([query])[0]
        self.entries.append(
            CacheEntry(
                query=query,
                response=response,
                embedding=embedding,
                mode=mode,
                expires_at=time.time() + ttl,
            )
        )
