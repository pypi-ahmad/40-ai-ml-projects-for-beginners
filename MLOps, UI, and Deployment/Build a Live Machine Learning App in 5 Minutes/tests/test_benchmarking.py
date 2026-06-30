"""Tests for benchmark export and reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmarking import BenchmarkRunner, format_benchmark_table
from src.schemas import BenchmarkResult


def _sample_result(model: str, prompt_key: str, latency: float) -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        prompt_key=prompt_key,
        runs=2,
        successful_runs=2,
        mean_latency_ms=latency,
        p95_latency_ms=latency + 10.0,
        mean_tokens_per_sec=45.0,
        mean_memory_mb=3072.0,
        cold_start_latency_ms=latency + 20.0,
        warm_start_latency_ms=latency,
        quality_score=78.0,
        error=None,
    )


def test_export_bundle_writes_consistent_artifacts(tmp_path: Path) -> None:
    results_by_prompt = {
        "short": [_sample_result("qwen3.5:2b", "short", 110.0)],
        "medium": [_sample_result("qwen3.5:2b", "medium", 140.0)],
        "long": [_sample_result("qwen3.5:2b", "long", 190.0)],
    }

    manifest = BenchmarkRunner.export_bundle(
        results_by_prompt=results_by_prompt,
        output_dir=str(tmp_path),
        primary_prompt="medium",
    )

    required_keys = {
        "short_json",
        "short_csv",
        "medium_json",
        "medium_csv",
        "long_json",
        "long_csv",
        "benchmark_results_json",
        "benchmark_results_csv",
        "benchmark_table",
        "artifact_manifest",
    }
    assert required_keys.issubset(set(manifest))

    for key in required_keys:
        assert Path(manifest[key]).exists(), f"Missing artifact for key={key}"

    artifact_manifest = json.loads(Path(manifest["artifact_manifest"]).read_text(encoding="utf-8"))
    assert artifact_manifest["primary_prompt"] == "medium"
    assert "generated_at" in artifact_manifest


def test_format_benchmark_table_has_required_columns() -> None:
    table = format_benchmark_table([_sample_result("qwen3.5:2b", "medium", 123.0)])
    assert "Cold (ms)" in table
    assert "Warm (ms)" in table
    assert "Quality" in table
