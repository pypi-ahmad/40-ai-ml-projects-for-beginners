from __future__ import annotations

from pathlib import Path

from reasoning_agent.settings import load_settings


def test_env_overrides_yaml(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
models:
  primary: qwen3:8b
benchmark:
  dataset_path: benchmarks/prompts.jsonl
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("PRIMARY_MODEL", "granite4.1:3b")
    monkeypatch.setenv("BENCHMARK_DATASET_PATH", "benchmarks/smoke_prompts.jsonl")

    load_settings.cache_clear()
    settings = load_settings(cfg)

    assert settings.primary_model == "granite4.1:3b"
    assert settings.benchmark_dataset_path == "benchmarks/smoke_prompts.jsonl"
