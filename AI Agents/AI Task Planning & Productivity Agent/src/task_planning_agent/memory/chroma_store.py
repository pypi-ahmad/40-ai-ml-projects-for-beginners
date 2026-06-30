"""ChromaDB semantic memory store."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb


def _naive_embedding(text: str, dim: int = 32) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [int(b) / 255.0 for b in digest[:dim]]
    return values


class ChromaSemanticStore:
    """Persist and search semantic plan/task memories."""

    def __init__(self, persist_dir: str, collection_name: str = "plan_memory") -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def upsert(self, item_id: str, text: str, metadata: dict[str, Any]) -> None:
        self.collection.upsert(
            ids=[item_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[_naive_embedding(text)],
        )

    def query(self, query_text: str, n_results: int = 8) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[_naive_embedding(query_text)],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        out: list[dict[str, Any]] = []
        for doc, meta, distance in zip(docs, metas, dists, strict=False):
            out.append({"document": doc, "metadata": meta, "distance": distance})
        return out
