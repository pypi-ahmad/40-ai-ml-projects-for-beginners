"""Indexing service for persistent ChromaDB with incremental updates."""

from __future__ import annotations

import time
from dataclasses import dataclass

from hybrid_research_assistant.chunking import Chunker
from hybrid_research_assistant.embeddings import EmbeddingProvider
from hybrid_research_assistant.index_manifest import IndexManifestManager
from hybrid_research_assistant.loaders import DocumentLoader
from hybrid_research_assistant.schemas import ChunkingStrategy
from hybrid_research_assistant.vectordb import ChromaVectorStore


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
    """Build or incrementally update Chroma index."""

    def __init__(
        self,
        loader: DocumentLoader,
        embedder: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        manifest: IndexManifestManager,
        *,
        schema_version: str = "1.0",
        collection_name: str,
        namespace: str,
    ) -> None:
        self.loader = loader
        self.embedder = embedder
        self.vector_store = vector_store
        self.manifest = manifest
        self.schema_version = schema_version
        self.collection_name = collection_name
        self.namespace = namespace

    def build_or_update(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        strategy: ChunkingStrategy,
        force_rebuild: bool = False,
    ) -> IndexReport:
        """Index documents and upsert changed chunks only."""

        started_total = time.perf_counter()
        docs = self.loader.load_directory()
        if not docs:
            raise ValueError("No supported documents found in configured directories")

        chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy)
        all_chunks = chunker.split_documents(docs)

        embedding_dimension = self.embedder.embedding_dimension()
        diff = self.manifest.diff(
            docs,
            schema_version=self.schema_version,
            collection_name=self.collection_name,
            namespace=self.namespace,
            embedding_model=self.embedder.model_name,
            chunking_strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dimension=embedding_dimension,
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
            started_embedding = time.perf_counter()
            embeddings = self.embedder.embed_texts([chunk.text for chunk in target_chunks])
            embedding_ms = (time.perf_counter() - started_embedding) * 1000
            self.vector_store.upsert_chunks(target_chunks, embeddings)

        self.manifest.save(
            docs,
            schema_version=self.schema_version,
            collection_name=self.collection_name,
            namespace=self.namespace,
            embedding_model=self.embedder.model_name,
            chunking_strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dimension=embedding_dimension,
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
