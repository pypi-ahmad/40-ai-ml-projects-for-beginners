"""Project-wide configuration and settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "project21_textclf"


class PathConfig(BaseModel):
    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")
    cache_dir: Path = Path(".cache")


class DatasetConfig(BaseModel):
    primary: str = "setfit_20_newsgroups"
    additional: list[str] = Field(default_factory=lambda: ["ag_news", "imdb"])
    validation_size: float = 0.1
    test_size: float = 0.2


class ModelConfig(BaseModel):
    required: list[str] = Field(
        default_factory=lambda: [
            "distilbert",
            "bert_base",
            "roberta_base",
            "deberta_v3_base",
            "modernbert",
        ]
    )
    optional: list[str] = Field(default_factory=lambda: ["minilm"])
    strategies: dict[str, str] = Field(
        default_factory=lambda: {
            "distilbert": "full",
            "bert_base": "full",
            "roberta_base": "lora",
            "deberta_v3_base": "lora",
            "modernbert": "lora",
        }
    )


class TrainingConfig(BaseModel):
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.06
    max_length: int = 256
    gradient_checkpointing: bool = True
    fp16: bool = True
    early_stopping_patience: int = 2
    save_total_limit: int = 2
    metric_for_best_model: str = "eval_macro_f1"


class HPOConfig(BaseModel):
    enabled: bool = True
    n_trials: int = 20
    timeout_sec: int = 12 * 60 * 60


class OptimizationConfig(BaseModel):
    export_onnx: bool = True
    dynamic_quantization: bool = True
    opset: int = 17


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    top_k: int = 3
    model_cache_size: int = 2048


class NotebookConfig(BaseModel):
    mode: str = "full"
    run_benchmarks: bool = True


class AppConfig(BaseSettings):
    """Top-level app configuration."""

    model_config = SettingsConfigDict(env_prefix="TCFW_", env_nested_delimiter="__", extra="ignore")

    mode: str = "full"
    seed: int = 42
    device: str = "cuda"
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    notebook: NotebookConfig = Field(default_factory=NotebookConfig)


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file merged with env vars.

    Args:
        path: Optional YAML config path.

    Returns:
        Resolved app config object.
    """
    if path is None:
        return AppConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        yaml_payload = yaml.safe_load(file) or {}

    defaults = AppConfig().model_dump()
    merged = _deep_update(defaults, yaml_payload)
    return AppConfig(**merged)
