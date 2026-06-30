"""Application configuration for enterprise document Q&A system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseModel):
    """Chunking defaults and experiment grid."""

    default_chunk_size: int = 768
    default_chunk_overlap: int = 100
    chunk_sizes: list[int] = Field(default_factory=lambda: [256, 512, 768, 1024, 1500])
    chunk_overlaps: list[int] = Field(default_factory=lambda: [0, 50, 100, 200])


class RetrievalSettings(BaseModel):
    """Retriever defaults and strategy controls."""

    default_k: int = 5
    candidate_ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    default_strategy: Literal["vector", "keyword", "hybrid"] = "hybrid"
    hybrid_rrf_k: int = 60
    hybrid_vector_weight: float = 0.55


class PromptSettings(BaseModel):
    """Prompt policy controls for grounded answering."""

    default_template: Literal[
        "strict_grounding",
        "citation_focus",
        "enterprise_qa",
        "legal_qa",
        "technical_qa",
        "unknown_safe",
    ] = "enterprise_qa"
    strict_grounding: bool = True
    unavailable_response: str = "Information unavailable in provided context."
    citation_instruction: str = (
        "Cite every factual claim using [source_path#chunk_id] markers."
    )


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:4b"
    generation_model: str = "qwen3.5:4b"
    judge_model: str = "granite4.1:3b"

    data_dir: Path = Path("data")
    documents_dir: Path = Path("data/documents")
    quickstart_documents_dir: Path = Path("data/documents_quickstart")
    quickstart_seed_documents_dir: Path = Path("data/documents_curated")
    eval_dir: Path = Path("data/eval")

    vector_db_path: Path = Path("vectordb/chroma")
    lexical_index_path: Path = Path("vectordb/bm25_index.json")

    outputs_dir: Path = Path("outputs")
    reports_dir: Path = Path("outputs/reports")
    benchmarks_dir: Path = Path("outputs/benchmarks")
    diagrams_dir: Path = Path("outputs/diagrams")
    visualizations_dir: Path = Path("outputs/visualizations")
    screenshots_dir: Path = Path("outputs/screenshots")
    corpus_manifest_path: Path = Path("outputs/reports/corpus_manifest.json")

    index_manifest_path: Path = Path("vectordb/index_manifest.json")
    collection_name: str = "enterprise_document_qa"
    corpus_profile: Literal["full", "quickstart"] = "full"
    duplicate_policy: Literal["skip_exact", "keep_all"] = "skip_exact"
    embedding_batch_size: int = 32

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)

    generation_temperature: float = 0.1
    generation_max_tokens: int = 700
    generation_streaming_default: bool = True

    def ensure_directories(self) -> None:
        """Create expected project directories."""

        for directory in (
            self.data_dir,
            self.documents_dir,
            self.quickstart_documents_dir,
            self.quickstart_seed_documents_dir,
            self.eval_dir,
            self.vector_db_path,
            self.outputs_dir,
            self.reports_dir,
            self.benchmarks_dir,
            self.diagrams_dir,
            self.visualizations_dir,
            self.screenshots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.lexical_index_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def active_documents_dir(self) -> Path:
        """Return document directory for selected corpus profile."""

        if self.corpus_profile == "quickstart":
            return self.quickstart_documents_dir
        return self.documents_dir

    @property
    def active_collection_name(self) -> str:
        """Return collection name scoped by corpus profile."""

        return f"{self.collection_name}_{self.corpus_profile}"

    @property
    def active_index_manifest_path(self) -> Path:
        """Return manifest path scoped by corpus profile."""

        return self.index_manifest_path.with_name(
            f"{self.index_manifest_path.stem}_{self.corpus_profile}.json"
        )

    @property
    def active_lexical_index_path(self) -> Path:
        """Return lexical BM25 index path scoped by corpus profile."""

        return self.lexical_index_path.with_name(
            f"{self.lexical_index_path.stem}_{self.corpus_profile}.json"
        )
