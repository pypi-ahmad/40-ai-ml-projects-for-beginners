"""Persistent ChromaDB integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from hybrid_research_assistant.schemas import ChunkRecord, RetrievedContext


class ChromaVectorStore:
    """Manage persistent Chroma collection lifecycle and CRUD operations."""

    def __init__(self, db_path: Path, collection_name: str) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate active collection."""

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Insert/update chunk vectors and metadata."""

        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """Delete vectors by doc id filter."""

        if not doc_ids:
            return
        for doc_id in doc_ids:
            self.collection.delete(where={"doc_id": doc_id})

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedContext]:
        """Similarity search with optional metadata filtering."""

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedContext] = []
        for chunk_id, text, metadata, distance in zip(ids, docs, metadatas, distances, strict=False):
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            row = dict(metadata)
            retrieved.append(
                RetrievedContext(
                    chunk_id=chunk_id,
                    doc_id=str(row.get("doc_id", "")),
                    text=text,
                    score=score,
                    metadata=row,
                    source="local",
                )
            )
        return retrieved

    def count(self) -> int:
        """Return number of indexed vectors."""

        return self.collection.count()

    def list_documents(self) -> list[dict[str, Any]]:
        """List unique indexed source docs."""

        payload = self.collection.get(include=["metadatas"])
        rows = payload.get("metadatas") or []
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            source = str(row.get("source", ""))
            unique[source] = {
                "source": source,
                "source_type": row.get("source_type"),
                "document_hash": row.get("document_hash"),
                "namespace": row.get("namespace"),
            }
        return sorted(unique.values(), key=lambda item: item["source"])
