"""Configuration loading from YAML and environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathSettings(BaseModel):
    data_dir: Path = Path("data")
    documents_dir: Path = Path("data/documents")
    notes_dir: Path = Path("data/notes")
    eval_dir: Path = Path("data/eval")
    web_cache_dir: Path = Path("data/web_cache")
    vector_db_dir: Path = Path("vectordb/chroma")
    manifest_path: Path = Path("vectordb/index_manifest.json")
    outputs_dir: Path = Path("outputs")
    logs_dir: Path = Path("outputs/logs")


class ModelSettings(BaseModel):
    primary_llm: str = "qwen3.5:4b"
    judge_llm: str = "granite4.1:3b"
    embedding_default: str = "BAAI/bge-small-en-v1.5"
    embedding_candidates: list[str] = Field(
        default_factory=lambda: [
            "BAAI/bge-small-en-v1.5",
            "all-MiniLM-L6-v2",
            "nomic-embed-text",
        ]
    )
    reranker_model: str = "BAAI/bge-reranker-base"
    ocr_model: str = "glm-ocr:latest"


class ChunkingSettings(BaseModel):
    strategy_default: str = "recursive"
    chunk_size_default: int = 768
    chunk_overlap_default: int = 100
    chunk_sizes: list[int] = Field(default_factory=lambda: [256, 512, 768, 1024])
    chunk_overlaps: list[int] = Field(default_factory=lambda: [0, 50, 100, 200])
    token_chunk_size_default: int = 512


class RetrievalSettings(BaseModel):
    top_k_default: int = 5
    candidate_k_default: int = 20
    mmr_lambda: float = 0.5
    similarity_threshold: float = 0.25
    route_auto: bool = True


class WebSettings(BaseModel):
    provider_default: Literal["duckduckgo", "tavily", "brave"] = "duckduckgo"
    providers_enabled: list[str] = Field(default_factory=lambda: ["duckduckgo", "tavily", "brave"])
    max_results: int = 8
    timeout_seconds: int = 5
    retry_attempts: int = 3
    cache_ttl_seconds: int = 900
    freshness_days: int = 7


class MemorySettings(BaseModel):
    max_turns: int = 20
    summary_trigger_turns: int = 12
    summary_max_tokens: int = 256


class CacheSettings(BaseModel):
    semantic_enabled: bool = True
    semantic_similarity_threshold: float = 0.92
    semantic_ttl_seconds: int = 1800
    response_ttl_local_seconds: int = 3600
    response_ttl_web_seconds: int = 600


class GenerationSettings(BaseModel):
    temperature: float = 0.1
    max_tokens: int = 800
    grounded_fallback: str = "I don't know based on the retrieved information."


class EvaluationSettings(BaseModel):
    benchmark_path: Path = Path("data/eval/benchmark_questions.jsonl")
    reports_dir: Path = Path("outputs/reports")


class UISettings(BaseModel):
    page_title: str = "Hybrid Real-Time AI Research Assistant"
    dark_mode_default: bool = True


class RuntimeSettings(BaseModel):
    profile: Literal["quickstart", "full"] = "quickstart"
    gpu_optimized: bool = True
    seed: int = 42


class IndexingSettings(BaseModel):
    collection_name: str = "hybrid_research_assistant"
    namespace: str = "default"
    incremental: bool = True
    deduplicate_chunks: bool = True
    normalize_embeddings: bool = True
    embedding_batch_size: int = 32


class AppSettings(BaseSettings):
    """Top-level app settings with env override support."""

    model_config = SettingsConfigDict(env_prefix="HRA_", env_file=".env", extra="ignore")

    ollama_host: str = "http://127.0.0.1:11434"
    profile: Literal["quickstart", "full"] = "quickstart"
    enable_analytics: bool = False

    models: ModelSettings = Field(default_factory=ModelSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    web_search: WebSettings = Field(default_factory=WebSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    ui: UISettings = Field(default_factory=UISettings)

    @property
    def active_collection_name(self) -> str:
        return f"{self.indexing.collection_name}_{self.profile}"

    @property
    def active_manifest_path(self) -> Path:
        stem = self.paths.manifest_path.stem
        return self.paths.manifest_path.with_name(f"{stem}_{self.profile}.json")

    def ensure_directories(self) -> None:
        """Create required runtime directories."""

        for path in (
            self.paths.data_dir,
            self.paths.documents_dir,
            self.paths.notes_dir,
            self.paths.eval_dir,
            self.paths.web_cache_dir,
            self.paths.vector_db_dir,
            self.paths.outputs_dir,
            self.paths.logs_dir,
            self.evaluation.reports_dir,
            self.paths.outputs_dir / "benchmarks",
            self.paths.outputs_dir / "diagrams",
            self.paths.outputs_dir / "screenshots",
        ):
            path.mkdir(parents=True, exist_ok=True)


def _yaml_to_raw(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime", {})
    return {
        "models": payload.get("models", {}),
        "runtime": runtime,
        "paths": payload.get("paths", {}),
        "indexing": payload.get("indexing", {}),
        "chunking": payload.get("chunking", {}),
        "retrieval": payload.get("retrieval", {}),
        "web_search": payload.get("web_search", {}),
        "memory": payload.get("memory", {}),
        "cache": payload.get("cache", {}),
        "generation": payload.get("generation", {}),
        "evaluation": payload.get("evaluation", {}),
        "ui": payload.get("ui", {}),
        "profile": runtime.get("profile", "quickstart"),
    }


@lru_cache(maxsize=1)
def load_settings(config_path: str | Path = "configs/app.yaml") -> AppSettings:
    """Load settings from YAML and apply env precedence."""

    base: dict[str, Any] = {}
    path = Path(config_path)
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = _yaml_to_raw(loaded)

    if "HRA_PROFILE" in os.environ:
        base.pop("profile", None)

    settings = AppSettings(**base)
    settings.ensure_directories()
    return settings
