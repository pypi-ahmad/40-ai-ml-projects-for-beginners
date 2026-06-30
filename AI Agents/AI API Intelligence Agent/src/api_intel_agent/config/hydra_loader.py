"""Hydra-based config composition entrypoint."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from api_intel_agent.config.settings import AppConfigModel


def load_with_hydra(config_dir: str = "configs", config_name: str = "settings") -> AppConfigModel:
    config_path = Path(config_dir).resolve()
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name=config_name)
        data = OmegaConf.to_container(cfg, resolve=True)
    return AppConfigModel.model_validate(data)
