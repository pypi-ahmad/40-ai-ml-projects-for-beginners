"""Chroma-backed semantic memory with graceful fallback."""

from __future__ import annotations

import os
from typing import Any

try:
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None

from api_intel_agent.config import load_settings
from api_intel_agent.core.schemas import MemorySearchHit


class ChromaMemoryStore:
    def __init__(self) -> None:
        settings = load_settings()
        self.enabled = bool(
            settings.memory.chroma_enabled
            and chromadb is not None
            and os.getenv("AGENT_DISABLE_CHROMA", "0") != "1"
        )
        self.collection_name = "analysis_memory"
        self.client = None
        self.collection = None

        if self.enabled:
            self.client = chromadb.PersistentClient(path=settings.memory.chroma_path)
            self.collection = self.client.get_or_create_collection(self.collection_name)

    def upsert(self, record_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.collection is None:
            return
        try:
            self.collection.upsert(ids=[record_id], documents=[content], metadatas=[metadata or {}])
        except Exception:
            # Graceful degradation for restricted environments (offline/model cache read-only).
            self.enabled = False

    def search(self, query: str, top_k: int = 5) -> list[MemorySearchHit]:
        if not self.enabled or self.collection is None:
            return []

        try:
            result = self.collection.query(query_texts=[query], n_results=top_k)
        except Exception:
            self.enabled = False
            return []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        hits: list[MemorySearchHit] = []
        for idx, doc, distance, metadata in zip(ids, documents, distances, metadatas, strict=False):
            score = 1.0 / (1.0 + float(distance or 0.0))
            hits.append(
                MemorySearchHit(
                    id=str(idx),
                    score=score,
                    content=str(doc),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return hits
