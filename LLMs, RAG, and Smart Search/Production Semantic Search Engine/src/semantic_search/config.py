"""Config loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EmbeddingModelConfig(BaseModel):
    provider: str
    model_name: str
    batch_size: int = 32
    normalize: bool = True


class EmbeddingConfig(BaseModel):
    primary: EmbeddingModelConfig
    comparisons: list[EmbeddingModelConfig] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    sentence_similarity_threshold: float = 0.72
    tokenizer: str = "cl100k_base"


class DatasetConfig(BaseModel):
    source: str = "huggingface"
    hf_dataset_id: str = "khalidalt/HuffPost"
    hf_fallback_dataset_ids: list[str] = Field(default_factory=lambda: ["heegyu/news-category-dataset"])
    split: str = "train"
    sample_size: int = 50000
    seed: int = 42
    enable_article_enrichment: bool = True
    enrichment_sample_size: int = 2000
    request_timeout_seconds: int = 7
    max_article_chars: int = 8000
    max_concurrent_enrichment: int = 8
    allowed_url_domains: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    max_document_chars: int = 25000
    min_document_chars: int = 20
    detect_language: bool = True
    deduplicate: bool = True
    normalize_whitespace: bool = True


class RetrievalConfig(BaseModel):
    vector_metric: str = "cosine"
    top_k: int = 10
    vector_candidates: int = 50
    mmr_lambda: float = 0.6
    mmr_enabled: bool = True
    hybrid_enabled: bool = True
    rrf_k: int = 60
    similarity_threshold: float = 0.05


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-base"
    top_n: int = 25


class VectorDbConfig(BaseModel):
    primary: str = "chroma"
    collection_name: str = "semantic_collection"
    namespace: str = "default"
    persist: bool = True


class EvaluationConfig(BaseModel):
    query_count: int = 150
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    llm_judge_model: str = "granite4.1:3b"
    use_llm_judge: bool = True
    baseline: str = "bm25"
    quality_lift_metric: str = "ndcg_at_10"
    min_relative_lift: float = 0.05


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = True
    file_name: str = "semantic_search.log"


class SecurityConfig(BaseModel):
    allow_local_files_only: bool = True
    max_upload_mb: int = 20
    block_hidden_files: bool = True
    readonly_search_mode: bool = True


class StreamlitConfig(BaseModel):
    page_title: str = "Semantic Search Engine"
    max_recent_searches: int = 30
    rate_limit_per_minute: int = 40


class AppConfig(BaseModel):
    project: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, str]
    dataset: DatasetConfig
    pipeline: PipelineConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    retrieval: RetrievalConfig
    vector_db: VectorDbConfig
    evaluation: EvaluationConfig
    streamlit: StreamlitConfig
    logging: LoggingConfig
    security: SecurityConfig
    query_processing: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)

    def path(self, key: str) -> Path:
        """Return path in config, resolved from repo root."""
        return Path(self.paths[key]).resolve()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load application config from YAML.

    Args:
        config_path: Optional path to config file.

    Returns:
        AppConfig parsed object.
    """
    env_path = os.getenv("SEMANTIC_SEARCH_CONFIG")
    target = Path(config_path) if config_path else Path(env_path) if env_path else Path("config/default.yaml")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    # Lightweight .env-style overrides for key runtime toggles.
    if dataset_id := os.getenv("HF_DATASET_ID"):
        raw.setdefault("dataset", {})["hf_dataset_id"] = dataset_id
    if split := os.getenv("HF_DATASET_SPLIT"):
        raw.setdefault("dataset", {})["split"] = split
    return AppConfig.model_validate(raw)
