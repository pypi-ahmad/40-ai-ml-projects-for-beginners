"""Application composition helpers."""

from __future__ import annotations

from pathlib import Path

from local_rag.config import AppSettings
from local_rag.embeddings import OllamaEmbeddingClient
from local_rag.generator import OllamaGenerator
from local_rag.index_manager import IndexManifestManager
from local_rag.indexing import IndexingService
from local_rag.loaders import DocumentLoader
from local_rag.rag import RAGPipeline
from local_rag.retriever import Retriever
from local_rag.vectordb import ChromaVectorStore


class AppRuntime:
    """Container for reusable runtime services."""

    def __init__(self, settings: AppSettings, generation_model: str | None = None) -> None:
        self.settings = settings
        settings.ensure_directories()

        self.loader = DocumentLoader(
            base_dir=settings.active_documents_dir,
            duplicate_policy=settings.duplicate_policy,
        )
        self.embedder = OllamaEmbeddingClient(
            model=settings.embedding_model,
            host=settings.ollama_host,
            normalize=settings.normalize_embeddings,
        )
        self.vector_store = ChromaVectorStore(
            db_path=settings.vector_db_path,
            collection_name=settings.active_collection_name,
        )
        self.manifest = IndexManifestManager(manifest_path=settings.active_index_manifest_path)
        self.indexer = IndexingService(
            settings=settings,
            loader=self.loader,
            embedder=self.embedder,
            vector_store=self.vector_store,
            manifest_manager=self.manifest,
        )
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.generator = OllamaGenerator(
            model=generation_model or settings.generation_model,
            host=settings.ollama_host,
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
        )
        self.pipeline = RAGPipeline(
            self.retriever,
            self.generator,
            strict_grounding=settings.prompts.strict_grounding,
            unavailable_response=settings.prompts.unavailable_response,
            citation_instruction=settings.prompts.citation_instruction,
        )


def load_settings() -> AppSettings:
    """Load settings from environment and defaults."""

    return AppSettings()


def ensure_eval_file(path: Path) -> None:
    """Create starter retrieval eval set when missing."""

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                (
                    '{"query":"What is ACPI?","relevant_doc_ids":[],'
                    '"relevant_chunk_ids":[],"answer":null}'
                ),
                (
                    '{"query":"How does systemd describe integration tests?","relevant_doc_ids":[],'
                    '"relevant_chunk_ids":[],"answer":null}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
