from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChromaResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class ChromaStore:
    def __init__(self, path: str, collection_name: str = "mcp_memory", enabled: bool = True) -> None:
        self.path = path
        self.collection_name = collection_name
        self._client = None
        self._collection = None

        if not enabled:
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(name=collection_name)
        except Exception:
            self._client = None
            self._collection = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    def upsert(self, items: list[dict[str, Any]]) -> None:
        if not self.available:
            return
        ids = [item["id"] for item in items]
        docs = [item["text"] for item in items]
        metadata = [item.get("metadata", {}) for item in items]
        self._collection.upsert(ids=ids, documents=docs, metadatas=metadata)

    def search(self, query: str, top_k: int) -> list[ChromaResult]:
        if not self.available:
            return []

        response = self._collection.query(query_texts=[query], n_results=top_k)
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[ChromaResult] = []
        for idx, identifier in enumerate(ids):
            score = 1.0
            if idx < len(distances):
                score = 1.0 / (1.0 + float(distances[idx]))
            results.append(
                ChromaResult(
                    id=identifier,
                    text=docs[idx] if idx < len(docs) else "",
                    metadata=metadatas[idx] if idx < len(metadatas) else {},
                    score=score,
                )
            )
        return results
