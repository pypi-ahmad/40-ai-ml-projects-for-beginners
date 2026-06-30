"""Vector storage backends: ChromaDB primary, FAISS comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from semantic_search.logging_utils import get_logger
from semantic_search.schemas import DocumentChunk
from semantic_search.utils import ensure_dir

logger = get_logger()


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, ensure_ascii=False)
    return clean


class ChromaVectorStore:
    """Persistent ChromaDB vector store."""

    def __init__(self, persist_dir: str | Path, collection_name: str):
        import chromadb

        self.persist_dir = ensure_dir(persist_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def reset_collection(self) -> None:
        """Drop and recreate collection for a clean full rebuild."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def upsert(self, chunks: list[DocumentChunk], embeddings: np.ndarray, batch_size: int = 5000) -> None:
        """Upsert chunk embeddings and metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding count mismatch")

        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            subset = chunks[start:end]
            ids = [chunk.chunk_id for chunk in subset]
            metadatas = [
                _sanitize_metadata(
                    chunk.metadata
                    | {
                        "chunk_index": chunk.chunk_index,
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                    }
                )
                for chunk in subset
            ]
            docs = [chunk.text for chunk in subset]
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas,
                documents=docs,
            )

        logger.info(
            "chroma_upsert_complete",
            count=total,
            batch_size=batch_size,
            persist_dir=str(self.persist_dir),
        )

    def delete_by_document_ids(self, document_ids: list[str]) -> None:
        """Delete chunks by document IDs."""
        for doc_id in document_ids:
            self.collection.delete(where={"document_id": doc_id})

    def query(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, list]:
        """Query nearest chunks from Chroma."""
        response = self.collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return response

    def get_by_ids(self, ids: list[str]) -> dict[str, list]:
        """Get entries by IDs."""
        return self.collection.get(ids=ids, include=["documents", "metadatas"])

    def count(self) -> int:
        """Count records in collection."""
        return int(self.collection.count())


class FaissVectorStore:
    """Local FAISS index for comparison benchmarks."""

    def __init__(self, index_dir: str | Path, metric: str = "cosine"):
        self.index_dir = ensure_dir(index_dir)
        self.metric = metric
        self.index: faiss.Index | None = None
        self.id_map: list[str] = []
        self.metadata: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, str] = {}

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Build FAISS index and store metadata/documents."""
        if embeddings.size == 0:
            raise ValueError("Embeddings empty")

        dim = int(embeddings.shape[1])
        matrix = embeddings.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(matrix)
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(matrix)
        self.id_map = [chunk.chunk_id for chunk in chunks]
        self.metadata = {chunk.chunk_id: chunk.metadata for chunk in chunks}
        self.documents = {chunk.chunk_id: chunk.text for chunk in chunks}
        logger.info("faiss_build_complete", count=len(self.id_map), dim=dim)

    def save(self) -> None:
        """Persist FAISS index and sidecar maps."""
        if self.index is None:
            raise RuntimeError("FAISS index missing")
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        (self.index_dir / "id_map.json").write_text(json.dumps(self.id_map), encoding="utf-8")
        (self.index_dir / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        (self.index_dir / "documents.json").write_text(json.dumps(self.documents), encoding="utf-8")

    def load(self) -> None:
        """Load persisted FAISS index."""
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.id_map = json.loads((self.index_dir / "id_map.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((self.index_dir / "metadata.json").read_text(encoding="utf-8"))
        self.documents = json.loads((self.index_dir / "documents.json").read_text(encoding="utf-8"))

    def query(self, query_embeddings: np.ndarray, top_k: int) -> dict[str, list]:
        """Query FAISS index and return aligned fields with Chroma path."""
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        matrix = query_embeddings.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(matrix)
        distances, indices = self.index.search(matrix, top_k)

        all_docs: list[list[str]] = []
        all_metas: list[list[dict[str, Any]]] = []
        all_scores: list[list[float]] = []
        for row_idx, neighbors in enumerate(indices):
            docs_row: list[str] = []
            meta_row: list[dict[str, Any]] = []
            score_row: list[float] = []
            for col_idx, faiss_idx in enumerate(neighbors):
                if faiss_idx < 0 or faiss_idx >= len(self.id_map):
                    continue
                chunk_id = self.id_map[int(faiss_idx)]
                docs_row.append(self.documents.get(chunk_id, ""))
                meta = dict(self.metadata.get(chunk_id, {}))
                meta["chunk_id"] = chunk_id
                meta_row.append(meta)
                score_row.append(float(distances[row_idx][col_idx]))
            all_docs.append(docs_row)
            all_metas.append(meta_row)
            all_scores.append(score_row)

        return {
            "documents": all_docs,
            "metadatas": all_metas,
            "distances": all_scores,
        }
