"""Index build/update service for persistent ChromaDB."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from local_rag.config import AppSettings
from local_rag.embeddings import OllamaEmbeddingClient
from local_rag.index_manager import IndexManifestManager
from local_rag.loaders import DocumentLoader
from local_rag.splitter import TextChunker
from local_rag.vectordb import ChromaVectorStore


@dataclass(slots=True)
class IndexReport:
    """Indexing execution summary."""

    total_documents: int
    total_chunks: int
    embedded_chunks: int
    removed_documents: int
    vector_count: int
    indexing_ms: float
    embedding_ms: float
    mode: str


class IndexingService:
    """Build or update persistent vector index."""

    def __init__(
        self,
        settings: AppSettings,
        loader: DocumentLoader,
        embedder: OllamaEmbeddingClient,
        vector_store: ChromaVectorStore,
        manifest_manager: IndexManifestManager,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.embedder = embedder
        self.vector_store = vector_store
        self.manifest_manager = manifest_manager

    def build_or_update(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        force_rebuild: bool = False,
    ) -> IndexReport:
        """Create or incrementally update vector index."""

        started_total = time.perf_counter()
        docs = self.loader.load_directory(self.settings.active_documents_dir)

        if not docs:
            raise ValueError("No documents found. Populate data/documents first.")

        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = chunker.split_documents(docs)

        previous_manifest = self.manifest_manager.load()
        embedding_dimension = int(previous_manifest.get("embedding_dimension", 0))
        if embedding_dimension <= 0:
            embedding_dimension = self._safe_embedding_dimension()

        diff = self.manifest_manager.diff(
            docs,
            manifest_schema_version=self.settings.manifest_schema_version,
            corpus_profile=self.settings.corpus_profile,
            embedding_model=self.settings.embedding_model,
            normalize_embeddings=self.embedder.normalize,
            embedding_dimension=embedding_dimension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=self.settings.active_collection_name,
        )

        mode = "incremental"
        if force_rebuild or not diff.compatible:
            mode = "full_rebuild"
            self.vector_store.reset()
            target_doc_ids = {doc.doc_id for doc in docs}
            removed_doc_ids: list[str] = []
        else:
            target_doc_ids = set(diff.added_or_changed_doc_ids)
            removed_doc_ids = diff.removed_doc_ids

        if removed_doc_ids:
            self.vector_store.delete_by_doc_ids(removed_doc_ids)

        target_chunks = [chunk for chunk in all_chunks if chunk.doc_id in target_doc_ids]

        embedding_ms = 0.0
        if target_chunks:
            try:
                embeddings, embedding_ms = self.embedder.timed_embed(
                    [chunk.text for chunk in target_chunks],
                    batch_size=self.settings.embedding_batch_size,
                )
            except TypeError:
                embeddings, embedding_ms = self.embedder.timed_embed(
                    [chunk.text for chunk in target_chunks]
                )
            if embeddings:
                embedding_dimension = len(embeddings[0])
            self.vector_store.upsert_chunks(target_chunks, embeddings)

        self.manifest_manager.save(
            docs,
            manifest_schema_version=self.settings.manifest_schema_version,
            corpus_profile=self.settings.corpus_profile,
            embedding_model=self.settings.embedding_model,
            normalize_embeddings=self.embedder.normalize,
            embedding_dimension=embedding_dimension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=self.settings.active_collection_name,
        )

        indexing_ms = (time.perf_counter() - started_total) * 1000

        return IndexReport(
            total_documents=len(docs),
            total_chunks=len(all_chunks),
            embedded_chunks=len(target_chunks),
            removed_documents=len(removed_doc_ids),
            vector_count=self.vector_store.count(),
            indexing_ms=indexing_ms,
            embedding_ms=embedding_ms,
            mode=mode,
        )

    def _safe_embedding_dimension(self) -> int:
        """Get embedding dimension without breaking indexing when Ollama is unavailable."""

        if hasattr(self.embedder, "embedding_dimension"):
            try:
                return int(self.embedder.embedding_dimension())
            except Exception:  # noqa: BLE001
                return 0
        return 0


def report_to_dict(report: IndexReport) -> dict[str, Any]:
    """Convert index report to serializable dictionary."""

    return {
        "total_documents": report.total_documents,
        "total_chunks": report.total_chunks,
        "embedded_chunks": report.embedded_chunks,
        "removed_documents": report.removed_documents,
        "vector_count": report.vector_count,
        "indexing_ms": report.indexing_ms,
        "embedding_ms": report.embedding_ms,
        "mode": report.mode,
    }
