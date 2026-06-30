from pathlib import Path

from domain_llm_ft.config.loader import load_config


def test_load_config_baseline() -> None:
    cfg = load_config(Path("configs/baseline.yaml"))
    assert cfg.experiment_name == "domain_llm_ft"
    assert cfg.dataset.name == "ag_news"
    assert abs(sum(cfg.dataset.split_ratio) - 1.0) < 1e-8
