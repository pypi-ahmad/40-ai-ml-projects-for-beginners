"""ChromaDB-backed semantic memory."""

from __future__ import annotations

import hashlib
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from internet_agent.config import Settings


class ChromaMemoryStore:
    """Persistent vector store for retrieved docs and prior answers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = chromadb.PersistentClient(path=settings.memory.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.memory.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = self._load_embedder(settings.llm.embedding_model)
        self._fallback_dim = 64

    @staticmethod
    def _load_embedder(model_name: str) -> SentenceTransformer | None:
        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            return None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is not None:
            return self._embedder.encode(texts, normalize_embeddings=True).tolist()

        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            values = []
            for idx in range(self._fallback_dim):
                byte = digest[idx % len(digest)]
                values.append((byte / 255.0) * 2 - 1)
            norm = sum(v * v for v in values) ** 0.5 or 1.0
            vectors.append([v / norm for v in values])
        return vectors

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        embeddings = self._embed(texts)
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, text: str, top_k: int) -> list[dict[str, Any]]:
        emb = self._embed([text])[0]
        result = self._collection.query(query_embeddings=[emb], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        mets = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        out: list[dict[str, Any]] = []
        for idx, doc_id in enumerate(ids):
            out.append(
                {
                    "id": doc_id,
                    "content": docs[idx] if idx < len(docs) else "",
                    "metadata": mets[idx] if idx < len(mets) else {},
                    "distance": float(distances[idx]) if idx < len(distances) else 0.0,
                }
            )
        return out
