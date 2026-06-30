"""Chroma-backed semantic memory."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection


class ChromaSemanticStore:
    """Persistent semantic memory using ChromaDB."""

    def __init__(self, path: str, collection_name: str = "agent_memory") -> None:
        db_path = Path(path)
        db_path.mkdir(parents=True, exist_ok=True)
        chroma_cache = db_path / ".cache"
        chroma_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CHROMA_CACHE_DIR", str(chroma_cache))
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection: Collection = client.get_or_create_collection(name=collection_name)

    def add(self, item_id: str, text: str, metadata: dict[str, str] | None = None) -> None:
        self.collection.add(ids=[item_id], documents=[text], metadatas=[metadata or {}])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else []
        output: list[dict[str, str | float]] = []
        for idx, doc in enumerate(docs):
            output.append(
                {
                    "id": ids[idx] if idx < len(ids) else f"doc-{idx}",
                    "text": doc,
                    "distance": float(distances[idx]) if idx < len(distances) else 0.0,
                }
            )
        return output
