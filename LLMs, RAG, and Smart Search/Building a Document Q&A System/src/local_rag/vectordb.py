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

        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding length mismatch: {len(chunks)} vs {len(embeddings)}"
            )

        max_batch_size = int(self.client.get_max_batch_size())
        for start in range(0, len(chunks), max_batch_size):
            end = min(start + max_batch_size, len(chunks))
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                embeddings=batch_embeddings,
                metadatas=[chunk.metadata for chunk in batch_chunks],
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
            score = 1.0 - float(distance)
            retrieved.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    doc_id=str(metadata.get("doc_id", "")),
                    text=str(text),
                    score=score,
                    metadata=dict(metadata),
                    strategy="vector",
                    vector_score=score,
                )
            )
        return retrieved

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """Return chunk records for given ids preserving input order."""

        if not chunk_ids:
            return []
        payload = self.collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        ids = [str(row) for row in payload.get("ids", [])]
        documents = [str(row) for row in payload.get("documents", [])]
        metadatas = [dict(row) for row in payload.get("metadatas", [])]
        rows_by_id: dict[str, ChunkRecord] = {}
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False):
            rows_by_id[chunk_id] = ChunkRecord(
                chunk_id=chunk_id,
                doc_id=str(metadata.get("doc_id", "")),
                text=text,
                metadata=metadata,
            )
        return [rows_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in rows_by_id]

    def export_chunks(self, where: dict[str, Any] | None = None) -> list[ChunkRecord]:
        """Export all current chunks from collection for lexical indexing."""

        rows: list[ChunkRecord] = []
        for payload in self._paged_get(where=where, include=["documents", "metadatas"]):
            ids = [str(row) for row in payload.get("ids", [])]
            documents = [str(row) for row in payload.get("documents", [])]
            metadatas = [dict(row) for row in payload.get("metadatas", [])]
            for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False):
                rows.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=str(metadata.get("doc_id", "")),
                        text=text,
                        metadata=metadata,
                    )
                )
        return rows

    def list_indexed_documents(self) -> list[dict[str, Any]]:
        """Return unique indexed source metadata records."""

        unique: dict[str, dict[str, Any]] = {}
        for payload in self._paged_get(include=["metadatas"]):
            metadatas = payload.get("metadatas") or []
            for meta in metadatas:
                source = str(meta.get("source_path", ""))
                unique[source] = {
                    "source_path": source,
                    "document_name": meta.get("document_name"),
                    "source_type": meta.get("source_type"),
                    "section": meta.get("section"),
                    "version_id": meta.get("version_id"),
                }
        return sorted(unique.values(), key=lambda row: row["source_path"])

    def _paged_get(
        self,
        *,
        where: dict[str, Any] | None = None,
        include: list[str],
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch collection rows in pages to avoid oversized SQLite queries."""

        size = page_size or min(int(self.client.get_max_batch_size()), 1000)
        size = max(size, 1)
        offset = 0
        pages: list[dict[str, Any]] = []

        while True:
            payload = self.collection.get(
                where=where,
                include=include,
                limit=size,
                offset=offset,
            )
            ids = payload.get("ids", []) or []
            if not ids:
                break
            pages.append(payload)
            count = len(ids)
            offset += count
            if count < size:
                break

        return pages

    def count(self) -> int:
        """Return number of vectors in collection."""

        return self.collection.count()

    @property
    def raw_collection(self) -> Collection:
        """Expose raw Chroma collection for advanced operations and tests."""

        return self.collection
