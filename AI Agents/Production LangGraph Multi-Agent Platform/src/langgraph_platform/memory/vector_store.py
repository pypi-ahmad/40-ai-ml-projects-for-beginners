"""ChromaDB vector storage for semantic memory and RAG."""

from __future__ import annotations

import hashlib
import os
from typing import Any

try:
    import chromadb
except Exception:
    chromadb = None  # type: ignore[assignment]


class ChromaMemoryStore:
    """Persistent Chroma-backed vector memory."""

    def __init__(self, chroma_path: str, embed_model: str = "all-MiniLM-L6-v2") -> None:
        self._in_memory_records: list[dict[str, Any]] = []
        self.client = None
        self.collection = None
        if chromadb is not None:
            self.client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.client.get_or_create_collection(name="knowledge")
        self._embedder = None
        use_sentence_transformers = os.environ.get("USE_SENTENCE_TRANSFORMERS", "0") == "1"
        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(embed_model)
            except Exception:
                self._embedder = None

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = list(digest) * ((dim // len(digest)) + 1)
        return [(value / 255.0) for value in seed[:dim]]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is not None:
            return self._embedder.encode(texts, normalize_embeddings=True).tolist()
        return [self._hash_embedding(text) for text in texts]

    def add_documents(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]] | None = None
    ) -> None:
        """Embed and upsert documents."""

        embeddings = self._encode(documents)
        metadata_values = metadatas or [{} for _ in documents]
        if self.collection is not None:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata_values,
            )
            return

        for index, doc_id in enumerate(ids):
            self._in_memory_records.append(
                {
                    "id": doc_id,
                    "document": documents[index],
                    "embedding": embeddings[index],
                    "metadata": metadata_values[index],
                }
            )

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Semantic retrieval for query text."""

        embedding = self._encode([query])[0]
        if self.collection is not None:
            return self.collection.query(query_embeddings=[embedding], n_results=top_k)

        if not self._in_memory_records:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self._in_memory_records:
            distance = (
                sum(
                    abs(a - b)
                    for a, b in zip(embedding[:64], record["embedding"][:64], strict=False)
                )
                / 64
            )
            scored.append((distance, record))
        scored.sort(key=lambda item: item[0])
        top = scored[:top_k]
        return {
            "ids": [[item[1]["id"] for item in top]],
            "documents": [[item[1]["document"] for item in top]],
            "metadatas": [[item[1]["metadata"] for item in top]],
            "distances": [[item[0] for item in top]],
        }

    def close(self) -> None:
        """Close persistent client if backend supports it."""

        if self.client is None:
            return
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                return
