"""Configuration objects and loaders for project runtime.

The project exposes one dataclass-based config so notebooks, scripts,
and UI share exactly same runtime defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import tomllib

RuntimeProfile = Literal["fast", "balanced", "max_depth"]


@dataclass(slots=True)
class RAGConfig:
    """Main runtime configuration for local RAG project.

    Attributes are intentionally explicit to keep tutorial code readable.
    """

    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("chroma_db")
    artifacts_dir: Path = Path("data/artifacts")
    hf_cache_dir: Path = Path(".hf_cache")

    dataset_name: str = "rajpurkar/squad_v2"
    corpus_splits: tuple[str, ...] = ("train", "validation")
    eval_splits: tuple[str, ...] = ("validation",)
    embedding_model: str = "qwen3-embedding:4b"
    generator_model: str = "qwen3.5:4b"
    judge_model: str = "granite4.1:3b"
    ollama_host: str = "http://127.0.0.1:11434"

    chunk_size: int = 900
    chunk_overlap: int = 180
    parent_chunk_size: int = 1600
    parent_chunk_overlap: int = 200

    top_k: int = 6
    profile: RuntimeProfile = "max_depth"
    random_seed: int = 42
    sampling_seed: int = 42
    min_relevance_score: float = 0.40
    abstain_threshold: float = 0.25

    retrieval_eval_queries: int = 5000
    generation_eval_queries: int = 1000
    judge_eval_queries: int = 1000
    chunking_benchmark_docs: int = 8000
    enable_advanced_retrieval_eval: bool = True
    enable_judge_eval: bool = True
    reuse_processed_artifacts: bool = True

    def resolved(self) -> "RAGConfig":
        """Return config with resolved absolute paths."""
        return RAGConfig(
            **{
                **asdict(self),
                "project_root": self.project_root.resolve(),
                "data_dir": (self.project_root / self.data_dir).resolve(),
                "chroma_dir": (self.project_root / self.chroma_dir).resolve(),
                "artifacts_dir": (self.project_root / self.artifacts_dir).resolve(),
                "hf_cache_dir": (self.project_root / self.hf_cache_dir).resolve(),
            }
        )

    def apply_profile(self) -> "RAGConfig":
        """Return config adjusted for selected runtime profile."""
        cfg = self
        if self.profile == "fast":
            cfg = RAGConfig(
                **{
                    **asdict(self),
                    "corpus_splits": ("train", "validation"),
                    "eval_splits": ("validation",),
                    "retrieval_eval_queries": 400,
                    "generation_eval_queries": 120,
                    "judge_eval_queries": 120,
                    "chunking_benchmark_docs": 1200,
                }
            )
        elif self.profile == "balanced":
            cfg = RAGConfig(
                **{
                    **asdict(self),
                    "corpus_splits": ("train", "validation"),
                    "eval_splits": ("validation",),
                    "retrieval_eval_queries": 1500,
                    "generation_eval_queries": 400,
                    "judge_eval_queries": 400,
                    "chunking_benchmark_docs": 3000,
                }
            )
        elif self.profile == "max_depth":
            cfg = RAGConfig(
                **{
                    **asdict(self),
                    "corpus_splits": ("train", "validation"),
                    "eval_splits": ("validation",),
                }
            )

        return cfg.resolved()


def load_config(path: Path | str | None = None) -> RAGConfig:
    """Load configuration from TOML file.

    The function supports optional file overrides and keeps defaults when
    fields are omitted.
    """
    cfg = RAGConfig()
    if path is None:
        return cfg.apply_profile()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with file_path.open("rb") as f:
        data = tomllib.load(f)

    if "rag" not in data:
        return cfg.apply_profile()

    rag_data: dict[str, Any] = data["rag"]
    kwargs = asdict(cfg)
    for key, value in rag_data.items():
        if key in kwargs:
            if key in {"corpus_splits", "eval_splits"} and isinstance(value, list):
                kwargs[key] = tuple(str(v) for v in value)
                continue
            kwargs[key] = value

    return RAGConfig(**kwargs).apply_profile()
