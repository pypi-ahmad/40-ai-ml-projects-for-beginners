"""Index build/update service for persistent ChromaDB and lexical BM25."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from local_rag.config import AppSettings
from local_rag.embeddings import OllamaEmbeddingClient
from local_rag.index_manager import IndexManifestManager
from local_rag.lexical import BM25LexicalIndex
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
    lexical_chunk_count: int
    indexing_ms: float
    embedding_ms: float
    mode: str

    @property
    def chunks_per_second(self) -> float:
        seconds = self.indexing_ms / 1000.0 if self.indexing_ms > 0 else 0.0
        if seconds <= 0:
            return 0.0
        return self.total_chunks / seconds


class IndexingService:
    """Build or update persistent vector/lexical indexes."""

    def __init__(
        self,
        settings: AppSettings,
        loader: DocumentLoader,
        embedder: OllamaEmbeddingClient,
        vector_store: ChromaVectorStore,
        manifest_manager: IndexManifestManager,
        lexical_index: BM25LexicalIndex,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.embedder = embedder
        self.vector_store = vector_store
        self.manifest_manager = manifest_manager
        self.lexical_index = lexical_index

    def build_or_update(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        force_rebuild: bool = False,
    ) -> IndexReport:
        """Create or incrementally update vector and lexical index."""

        started_total = time.perf_counter()
        docs = self.loader.load_directory(self.settings.active_documents_dir)

        if not docs:
            raise ValueError("No documents found. Populate data/documents first.")

        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = chunker.split_documents(docs)

        diff = self.manifest_manager.diff(
            docs,
            embedding_model=self.settings.embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
            embeddings, embedding_ms = self.embedder.timed_embed(
                [chunk.text for chunk in target_chunks],
                batch_size=self.settings.embedding_batch_size,
            )
            self.vector_store.upsert_chunks(target_chunks, embeddings)

        # Keep lexical index in sync with current vector store content.
        lexical_chunks = self.vector_store.export_chunks()
        lexical_stats = self.lexical_index.build(lexical_chunks)

        self.manifest_manager.save(
            docs,
            embedding_model=self.settings.embedding_model,
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
            lexical_chunk_count=lexical_stats.chunk_count,
            indexing_ms=indexing_ms,
            embedding_ms=embedding_ms,
            mode=mode,
        )



def report_to_dict(report: IndexReport) -> dict[str, Any]:
    """Convert index report to serializable dictionary."""

    return {
        "total_documents": report.total_documents,
        "total_chunks": report.total_chunks,
        "embedded_chunks": report.embedded_chunks,
        "removed_documents": report.removed_documents,
        "vector_count": report.vector_count,
        "lexical_chunk_count": report.lexical_chunk_count,
        "indexing_ms": report.indexing_ms,
        "embedding_ms": report.embedding_ms,
        "chunks_per_second": report.chunks_per_second,
        "mode": report.mode,
    }
