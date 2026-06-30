"""Chroma vector storage utilities."""

from __future__ import annotations

from typing import Any

import chromadb

from multimodal_ai.domain import RetrievalHit


class ChromaStore:
    """Wrapper around persistent Chroma collections."""

    COLLECTIONS = {
        "image": "image_embeddings",
        "ocr": "ocr_embeddings",
        "document": "document_embeddings",
        "screenshot": "screenshot_embeddings",
        "chart": "chart_embeddings",
    }

    def __init__(self, path: str) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collections: dict[str, Any] = {}
        for key, name in self.COLLECTIONS.items():
            self._collections[key] = self._client.get_or_create_collection(name=name)

    def upsert(
        self,
        modality: str,
        record_id: str,
        vector: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert record into modality collection."""

        collection = self._collections[modality]
        collection.upsert(
            ids=[record_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[metadata],
        )

    def search(
        self, modality: str, query_vector: list[float], top_k: int = 5
    ) -> list[RetrievalHit]:
        """Search nearest neighbors by vector."""

        collection = self._collections[modality]
        response = collection.query(query_embeddings=[query_vector], n_results=top_k)

        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metadata_rows = response.get("metadatas", [[]])[0]

        hits: list[RetrievalHit] = []
        for item_id, distance, doc, metadata in zip(
            ids, distances, docs, metadata_rows, strict=False
        ):
            score = 1.0 / (1.0 + float(distance))
            hits.append(RetrievalHit(id=item_id, score=score, text=doc, metadata=metadata or {}))
        return hits
