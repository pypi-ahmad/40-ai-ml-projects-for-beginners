"""Configuration loading and runtime settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, Field


class RuntimeInfo(BaseModel):
    """Runtime metadata for model execution."""

    device: str = "cpu"
    gpu_available: bool = False
    cuda_version: str | None = None


class PlatformPaths(BaseModel):
    """Filesystem paths used by platform."""

    root: Path = Path(".")
    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    exports_dir: Path = Path("data/exports")
    chroma_dir: Path = Path("data/chroma")


class PlatformConfig(BaseModel):
    """Typed config subset used across services."""

    raw: dict[str, Any] = Field(default_factory=dict)
    app_name: str = "multimodal-ai-platform"
    sqlite_url: str = "sqlite:///./data/multimodal.db"
    chroma_path: str = "./data/chroma"
    default_vision_model: str = "qwen2_5_vl"
    default_llm_backend: str = "ollama"
    default_llm_model: str = "llama3"
    default_embedding_model: str = "clip"
    ocr_primary_engine: str = "glm_ocr"
    retrieval_top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 80
    seed: int = 42
    allow_network_download: bool = False


def _to_container(cfg: Any) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise ValueError("Hydra config root must be dict")
    return cast(dict[str, Any], container)


def load_config(path: str | Path = "configs/config.yaml") -> PlatformConfig:
    """Load YAML config tree into typed platform config."""

    cfg = OmegaConf.load(path)
    base = _to_container(cfg)

    models = base.get("models", {})
    ocr = base.get("ocr", {})
    retrieval = base.get("retrieval", {})
    storage = base.get("storage", {})
    runtime = base.get("runtime", {})
    app = base.get("app", {})

    return PlatformConfig(
        raw=base,
        app_name=app.get("name", "multimodal-ai-platform"),
        sqlite_url=storage.get("sqlite_url", "sqlite:///./data/multimodal.db"),
        chroma_path=storage.get("chroma_path", "./data/chroma"),
        default_vision_model=models.get("vision", {}).get("default", "qwen2_5_vl"),
        default_llm_backend=models.get("llm", {}).get("backend", "ollama"),
        default_llm_model=models.get("llm", {}).get("default_model", "llama3"),
        default_embedding_model=models.get("embeddings", {}).get("vision", "clip"),
        ocr_primary_engine=ocr.get("primary_engine", "glm_ocr"),
        retrieval_top_k=int(retrieval.get("top_k", 5)),
        chunk_size=int(retrieval.get("chunk_size", 500)),
        chunk_overlap=int(retrieval.get("chunk_overlap", 80)),
        seed=int(runtime.get("seed", 42)),
        allow_network_download=bool(runtime.get("allow_network_download", False)),
    )
