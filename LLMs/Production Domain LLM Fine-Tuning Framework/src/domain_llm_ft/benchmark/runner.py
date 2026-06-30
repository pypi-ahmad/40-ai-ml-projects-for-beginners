"""Model benchmarking utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from domain_llm_ft.models.registry import resolve_model_name


@dataclass
class BenchmarkResult:
    model: str
    latency_ms: float
    throughput: float
    memory_mb: float


def benchmark_model(model_name: str, samples: list[str], batch_size: int = 8) -> BenchmarkResult:
    """Benchmark model latency/throughput/memory on inference."""
    resolved = resolve_model_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(resolved, local_files_only=True)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    start = time.perf_counter()
    _ = classifier(samples, batch_size=batch_size, truncation=True)
    elapsed = time.perf_counter() - start
    throughput = len(samples) / max(elapsed, 1e-8)

    if torch.cuda.is_available():
        memory_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0.0

    return BenchmarkResult(
        model=resolved,
        latency_ms=(elapsed * 1000) / max(len(samples), 1),
        throughput=throughput,
        memory_mb=memory_mb,
    )


def benchmark_matrix(models: list[str], samples: list[str]) -> list[BenchmarkResult]:
    """Run benchmark for matrix of models."""
    results: list[BenchmarkResult] = []
    for model in models:
        try:
            results.append(benchmark_model(model, samples))
        except Exception as exc:  # pragma: no cover - env/cache dependent
            logger.warning("Skipping benchmark model {}: {}", model, exc)
    return results
