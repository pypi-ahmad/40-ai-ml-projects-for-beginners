from pathlib import Path

from llmft.config.loader import load_project_config


def test_config_loader_parses_nested_dataclasses() -> None:
    cfg = load_project_config(Path("configs/project.yaml"))
    assert cfg.runtime.artifacts_dir == "artifacts"
    assert cfg.inference.enable_remote_backends is False
    assert cfg.model.targets[0] == "llama3_8b"
