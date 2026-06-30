"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

from domain_llm_ft.config.schemas import ExperimentConfig
from domain_llm_ft.utils.io import read_yaml


def load_config(path: Path) -> ExperimentConfig:
    """Load experiment config from YAML.

    Args:
        path: Path to YAML file.

    Returns:
        Validated experiment config.

    Example:
        >>> cfg = load_config(Path("configs/baseline.yaml"))
        >>> cfg.model.name
        'distilbert-base-uncased'
    """
    return ExperimentConfig.model_validate(read_yaml(path))
