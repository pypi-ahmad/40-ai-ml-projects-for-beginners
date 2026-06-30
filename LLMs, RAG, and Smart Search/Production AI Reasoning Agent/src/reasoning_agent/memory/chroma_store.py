"""ChromaDB-backed semantic memory store with embedding fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.memory.base import MemoryEvent, MemoryHit, MemoryScope

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency runtime guard
    chromadb = None


class EmbeddingProvider:
    """Embedding provider with Ollama-first fallback to sentence-transformers."""

    def __init__(self, llm: OllamaClient, embedding_model: str) -> None:
        self.llm = llm
        self.embedding_model = embedding_model
        self._st_model: Any | None = None

    def embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""

        try:
            vec = self.llm.embed(self.embedding_model, text)
            if vec:
                return vec
        except Exception:
            pass

        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")

        vector = self._st_model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vector.tolist()]


class ChromaMemoryStore:
    """Persistent semantic memory with fallback to in-process vectors."""

    def __init__(
        self,
        chroma_dir: str,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "reasoning_agent_memory",
    ) -> None:
        self.embedding = embedding_provider
        self.collection_name = collection_name
        self._fallback_events: list[MemoryEvent] = []
        self._fallback_embeddings: list[list[float]] = []

        self.client = None
        self.collection = None
        if chromadb is not None:
            Path(chroma_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=chroma_dir)
            self.collection = self.client.get_or_create_collection(name=collection_name)

    def write(self, event: MemoryEvent) -> None:
        """Write memory event to Chroma or fallback store."""

        emb = self.embedding.embed(event.text)
        if self.collection is None:
            self._fallback_events.append(event)
            self._fallback_embeddings.append(emb)
            return

        item_id = f"{event.session_id}:{event.run_id}:{event.created_at.timestamp()}"
        metadata = {
            "session_id": event.session_id,
            "run_id": event.run_id,
            "scope": event.scope.value,
            **event.metadata,
        }
        self.collection.add(
            ids=[item_id],
            documents=[event.text],
            embeddings=[emb],
            metadatas=[metadata],
        )

    def retrieve(self, query: str, k: int = 5, scope: MemoryScope | None = None) -> list[MemoryHit]:
        """Retrieve semantically similar memory records."""

        q_emb = self.embedding.embed(query)

        if self.collection is None:
            return self._fallback_retrieve(q_emb, k, scope)

        where: dict[str, str] | None = None
        if scope is not None:
            where = {"scope": scope.value}

        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            where=where,
        )

        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        hits: list[MemoryHit] = []
        for doc, dist, meta in zip(docs, distances, metadatas, strict=False):
            score = 1.0 / (1.0 + float(dist))
            hits.append(MemoryHit(text=str(doc), score=score, metadata=dict(meta or {})))
        return hits

    def _fallback_retrieve(
        self,
        query_embedding: list[float],
        k: int,
        scope: MemoryScope | None,
    ) -> list[MemoryHit]:
        """Cosine similarity retrieval when Chroma unavailable."""

        if not self._fallback_embeddings:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        qnorm = np.linalg.norm(query)
        if qnorm == 0.0:
            return []

        scored: list[MemoryHit] = []
        for event, emb in zip(self._fallback_events, self._fallback_embeddings, strict=False):
            if scope is not None and event.scope != scope:
                continue

            vec = np.array(emb, dtype=np.float32)
            denom = np.linalg.norm(vec) * qnorm
            if denom == 0.0:
                continue
            score = float(np.dot(vec, query) / denom)
            scored.append(MemoryHit(text=event.text, score=score, metadata=event.metadata))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]
