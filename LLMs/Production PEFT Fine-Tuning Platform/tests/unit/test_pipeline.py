from __future__ import annotations

from pathlib import Path

from peft_platform.pipeline import load_config


def test_load_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("peft:\n  method: lora\ntrain:\n  max_steps: 1\nbenchmark:\n  runs: 1\ntracking:\n  mlflow_tracking_uri: sqlite:///tmp.db\n  experiment_name: test\n", encoding="utf-8")
    cfg = load_config(config_path)
    if isinstance(cfg, dict):
        assert str(cfg["peft"]["method"]) == "lora"
    else:
        assert str(cfg.peft.method) == "lora"
