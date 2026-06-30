"""Benchmark runner for local Ollama model performance comparisons."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.config import get_config
from src.ollama_client import OllamaClient
from src.schemas import BenchmarkResult

logger = logging.getLogger(__name__)


BENCHMARK_MODELS = [
    "qwen3.5:2b",
    "qwen3.5:4b",
    "granite4.1:3b",
    "nemotron-3-nano:4b",
]

BENCHMARK_PROMPTS = {
    "short": "Define machine learning in one paragraph.",
    "medium": (
        "Explain differences between supervised, unsupervised, and reinforcement learning "
        "with practical examples for each paradigm."
    ),
    "long": (
        "Write comprehensive explanation of transformer architecture, attention mechanism, "
        "training strategy, and deployment tradeoffs for edge and cloud inference."
    ),
}

QUALITY_RUBRIC_PROMPT = (
    "You are evaluator. Score model answer from 1 to 10 using relevance, correctness, and clarity. "
    "Return only integer score."
)

QUALITY_SCORE_PATTERN = re.compile(r"\b(10|[1-9])\b")


@dataclass
class BenchmarkArtifacts:
    """File paths produced by benchmark export."""

    json_path: str
    csv_path: str


class BenchmarkRunner:
    """Execute reproducible benchmark runs and export machine-readable outputs."""

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()
        self._owns_client = client is None

    def close(self) -> None:
        """Close owned client resources."""

        if self._owns_client:
            self.client.close()

    def _score_quality(self, candidate_answer: str) -> float:
        """Estimate answer quality via local judge model with deterministic fallback."""

        if not candidate_answer.strip():
            return 0.0

        judge_prompt = f"Prompted answer:\n{candidate_answer[:2000]}"
        judge_result = self.client.generate(
            model=get_config().chat_model,
            prompt=judge_prompt,
            system=QUALITY_RUBRIC_PROMPT,
            temperature=0.0,
            max_tokens=20,
        )

        if not judge_result["error"]:
            match = QUALITY_SCORE_PATTERN.search(judge_result["response"])
            if match:
                score = int(match.group(1))
                return round(score * 10.0, 2)

        word_count = len(candidate_answer.split())
        punctuation = sum(candidate_answer.count(ch) for ch in ".,;:")
        heuristic = min(100.0, max(20.0, word_count * 1.3 + punctuation * 2.0))
        return round(heuristic, 2)

    def run_single(
        self, model: str, prompt_key: str = "medium", runs: int | None = None
    ) -> BenchmarkResult:
        """Benchmark one model with repeated inference calls."""

        if prompt_key not in BENCHMARK_PROMPTS:
            prompt_key = "medium"

        resolved_runs = runs or get_config().benchmark_runs
        prompt = BENCHMARK_PROMPTS[prompt_key]

        stats = self.client.measure_inference_time(
            model=model,
            prompt=prompt,
            runs=resolved_runs,
            max_tokens=320,
        )

        quality_samples: list[float] = []
        for _ in range(min(2, resolved_runs)):
            sample = self.client.generate(
                model=model,
                prompt=prompt,
                temperature=0.2,
                max_tokens=320,
            )
            if sample["error"]:
                continue
            quality_samples.append(self._score_quality(sample["response"]))

        successful_runs = int(stats.get("successful_runs", 0))
        if quality_samples:
            quality_score = round(mean(quality_samples), 2)
        elif successful_runs > 0:
            quality_score = 50.0
        else:
            quality_score = 0.0

        latencies = [float(value) for value in stats.get("latencies_ms", [])]
        cold_start_latency_ms = latencies[0] if latencies else 0.0
        warm_start_latency_ms = (
            round(mean(latencies[1:]), 2) if len(latencies) > 1 else cold_start_latency_ms
        )

        return BenchmarkResult(
            model=model,
            prompt_key=prompt_key,
            runs=resolved_runs,
            successful_runs=successful_runs,
            mean_latency_ms=float(stats.get("mean_latency_ms", 0.0)),
            p95_latency_ms=float(stats.get("p95_latency_ms", 0.0)),
            mean_tokens_per_sec=float(stats.get("mean_tokens_per_sec", 0.0)),
            mean_memory_mb=float(stats.get("mean_memory_mb", 0.0)),
            cold_start_latency_ms=cold_start_latency_ms,
            warm_start_latency_ms=warm_start_latency_ms,
            quality_score=quality_score,
            error=stats.get("error"),
        )

    def run_all(
        self,
        prompt_key: str = "medium",
        runs: int | None = None,
        models: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Benchmark all configured models and return typed result list."""

        selected_models = models or BENCHMARK_MODELS
        results: list[BenchmarkResult] = []

        for model in selected_models:
            logger.info("Benchmarking model=%s prompt=%s", model, prompt_key)
            results.append(self.run_single(model=model, prompt_key=prompt_key, runs=runs))

        return results

    @staticmethod
    def _write_result_set(path: Path, results: list[BenchmarkResult]) -> tuple[Path, Path]:
        """Write one benchmark result set to JSON and CSV."""

        payload = [result.model_dump() for result in results]
        json_path = path.with_suffix(".json")
        csv_path = path.with_suffix(".csv")

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        pd.DataFrame(payload).to_csv(csv_path, index=False)
        return json_path, csv_path

    @staticmethod
    def export_results(
        results: list[BenchmarkResult], output_dir: str = "outputs/benchmarks"
    ) -> BenchmarkArtifacts:
        """Write benchmark results to JSON and CSV for notebooks and README."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        payload = [result.model_dump() for result in results]
        json_path = output_path / "benchmark_results.json"
        csv_path = output_path / "benchmark_results.csv"

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        pd.DataFrame(payload).to_csv(csv_path, index=False)

        return BenchmarkArtifacts(json_path=str(json_path), csv_path=str(csv_path))

    @staticmethod
    def export_bundle(
        results_by_prompt: dict[str, list[BenchmarkResult]],
        output_dir: str = "outputs/benchmarks",
        primary_prompt: str = "medium",
    ) -> dict[str, str]:
        """Export synchronized benchmark artifacts for all prompt groups."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, str] = {}
        for prompt_key, results in results_by_prompt.items():
            base = output_path / f"{prompt_key}_results"
            json_path, csv_path = BenchmarkRunner._write_result_set(base, results)
            manifest[f"{prompt_key}_json"] = str(json_path)
            manifest[f"{prompt_key}_csv"] = str(csv_path)

        primary_results = results_by_prompt.get(primary_prompt, [])
        primary_artifacts = BenchmarkRunner.export_results(primary_results, output_dir=output_dir)

        benchmark_table = format_benchmark_table(primary_results)
        table_path = output_path / "benchmark_table.md"
        table_path.write_text(benchmark_table, encoding="utf-8")

        manifest["benchmark_results_json"] = primary_artifacts.json_path
        manifest["benchmark_results_csv"] = primary_artifacts.csv_path
        manifest["benchmark_table"] = str(table_path)

        manifest_path = output_path / "artifact_manifest.json"
        manifest_payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "primary_prompt": primary_prompt,
            "files": manifest,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        manifest["artifact_manifest"] = str(manifest_path)

        return manifest


def format_benchmark_table(results: list[dict[str, Any]] | list[BenchmarkResult]) -> str:
    """Build markdown table used in terminal logs, notebook outputs, and README."""

    normalized: list[dict[str, Any]] = [
        item.model_dump() if isinstance(item, BenchmarkResult) else item for item in results
    ]

    header = (
        "| Model | Prompt | Mean Latency (ms) | P95 (ms) | Tokens/s | Memory (MB) | "
        "Cold (ms) | Warm (ms) | Quality | Error |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    rows = [header, sep]

    for row in normalized:
        rows.append(
            (
                "| {model} | {prompt_key} | {mean_latency_ms} | {p95_latency_ms} | "
                "{mean_tokens_per_sec} | {mean_memory_mb} | {cold_start_latency_ms} | "
                "{warm_start_latency_ms} | {quality_score} | {error} |"
            ).format(
                model=row.get("model", "N/A"),
                prompt_key=row.get("prompt_key", "N/A"),
                mean_latency_ms=row.get("mean_latency_ms", 0),
                p95_latency_ms=row.get("p95_latency_ms", 0),
                mean_tokens_per_sec=row.get("mean_tokens_per_sec", 0),
                mean_memory_mb=row.get("mean_memory_mb", 0),
                cold_start_latency_ms=row.get("cold_start_latency_ms", 0),
                warm_start_latency_ms=row.get("warm_start_latency_ms", 0),
                quality_score=row.get("quality_score", 0),
                error=row.get("error") or "",
            )
        )

    return "\n".join(rows)
