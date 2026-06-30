"""Persistent ChromaDB integration layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from local_rag.types import ChunkRecord, RetrievalResult


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
        """Drop and recreate collection."""

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reload(self) -> None:
        """Reload existing collection from persistent client."""

        self.collection = self.client.get_collection(self.collection_name)

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Insert or update chunk vectors and metadata."""

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """Delete vectors by document id filter."""

        if not doc_ids:
            return
        for doc_id in doc_ids:
            self.collection.delete(where={"doc_id": doc_id})

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run similarity search with optional metadata filter."""

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

        retrieved: list[RetrievalResult] = []
        for chunk_id, text, metadata, distance in zip(
            ids,
            docs,
            metadatas,
            distances,
            strict=False,
        ):
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=str(metadata.get("doc_id", "")),
                    text=text,
                    score=score,
                    metadata=dict(metadata),
                )
            )
        return retrieved

    def list_indexed_documents(self) -> list[dict[str, Any]]:
        """Return unique indexed source metadata records."""

        payload = self.collection.get(include=["metadatas"])
        metadatas = payload.get("metadatas") or []
        unique: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            source = str(meta.get("source_path", ""))
            unique[source] = {
                "source_path": source,
                "source_type": meta.get("source_type"),
                "section": meta.get("section"),
            }
        return sorted(unique.values(), key=lambda row: row["source_path"])

    def count(self) -> int:
        """Return number of vectors in collection."""

        return self.collection.count()

    def integrity_report(self) -> dict[str, int | float]:
        """Return basic collection integrity diagnostics."""

        payload = self.collection.get(include=["metadatas"])
        ids = payload.get("ids") or []
        metadatas = payload.get("metadatas") or []

        total_vectors = len(ids)
        duplicate_chunk_ids = max(0, total_vectors - len(set(ids)))
        missing_required_metadata = 0
        missing_doc_id = 0
        missing_source_path = 0
        missing_chunk_id_metadata = 0

        for meta in metadatas:
            has_doc_id = bool(meta.get("doc_id"))
            has_source_path = bool(meta.get("source_path"))
            has_chunk_id_meta = bool(meta.get("chunk_id"))
            if not has_doc_id:
                missing_doc_id += 1
            if not has_source_path:
                missing_source_path += 1
            if not has_chunk_id_meta:
                missing_chunk_id_metadata += 1
            if not (has_doc_id and has_source_path and has_chunk_id_meta):
                missing_required_metadata += 1

        return {
            "total_vectors": total_vectors,
            "duplicate_chunk_ids": duplicate_chunk_ids,
            "missing_required_metadata": missing_required_metadata,
            "missing_doc_id": missing_doc_id,
            "missing_source_path": missing_source_path,
            "missing_chunk_id_metadata": missing_chunk_id_metadata,
        }

    @property
    def raw_collection(self) -> Collection:
        """Expose raw Chroma collection for advanced operations and tests."""

        return self.collection
