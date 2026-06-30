"""Hydra/OmegaConf-backed configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    environment: str = "local_gpu"
    sqlite_path: str = "resume_ai.sqlite3"
    chroma_path: str = "chroma"
    max_workers: int = 8
    batch_size: int = 64


class ModelConfig(BaseModel):
    extraction_model: str = "llama3:8b"
    reasoning_model: str = "qwen3:8b"
    interview_model: str = "gemma3:4b"
    scoring_model: str = "deepseek-r1:8b"
    fallback_model: str = "phi4-mini:latest"
    parser_model: str = "mistral:7b"
    ollama_base_url: str = "http://localhost:11434"


class EmbeddingConfig(BaseModel):
    backend: str = "sentence_transformers"
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_embedding_model: str = "qwen3-embedding:4b"
    output_dimension: int = 384


class OCRConfig(BaseModel):
    enable_tesseract: bool = True
    enable_paddle: bool = False
    tesseract_lang: str = "eng"
    dpi: int = 300


class ScoringConfig(BaseModel):
    technical_skills: float = 0.30
    experience: float = 0.25
    projects: float = 0.20
    education: float = 0.10
    certifications: float = 0.05
    communication: float = 0.05
    bonus_skills: float = 0.05


class RetryConfig(BaseModel):
    llm_retries: int = 2
    timeout_seconds: int = 120


class AppConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)


def _deep_merge_dicts(*configs: DictConfig) -> dict[str, Any]:
    merged: DictConfig = OmegaConf.create()
    for cfg in configs:
        merged = OmegaConf.merge(merged, cfg)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def load_config(config_root: str | Path = "config") -> AppConfig:
    """Load full application config from YAML files.

    Args:
        config_root: Root directory containing config subfolders.

    Returns:
        AppConfig instance.
    """
    root = Path(config_root)
    default_cfg = OmegaConf.load(root / "config.yaml")
    runtime_cfg = OmegaConf.load(root / "runtime" / "default.yaml")
    model_cfg = OmegaConf.load(root / "models" / "default.yaml")
    emb_cfg = OmegaConf.load(root / "embeddings" / "default.yaml")
    ocr_cfg = OmegaConf.load(root / "ocr" / "default.yaml")
    scoring_cfg = OmegaConf.load(root / "scoring" / "default.yaml")
    retry_cfg = OmegaConf.load(root / "runtime" / "retries.yaml")

    merged = _deep_merge_dicts(
        default_cfg,
        OmegaConf.create({"runtime": runtime_cfg}),
        OmegaConf.create({"models": model_cfg}),
        OmegaConf.create({"embeddings": emb_cfg}),
        OmegaConf.create({"ocr": ocr_cfg}),
        OmegaConf.create({"scoring": scoring_cfg}),
        OmegaConf.create({"retries": retry_cfg}),
    )
    return AppConfig.model_validate(merged)
