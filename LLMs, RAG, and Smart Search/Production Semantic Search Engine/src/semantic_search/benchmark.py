"""Benchmark suite for embeddings and retrieval."""

from __future__ import annotations

import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from semantic_search.config import AppConfig, EmbeddingModelConfig
from semantic_search.embeddings import build_embedding_backend, embed_text_batches
from semantic_search.evaluation import EvaluationOutput, evaluate_retrieval
from semantic_search.schemas import BenchmarkRun, EvaluationCase


@dataclass(slots=True)
class EmbeddingBenchmarkRow:
    model_name: str
    provider: str
    dimension: int
    avg_latency_ms: float
    max_memory_mb: float


class BenchmarkRunner:
    """Run benchmark experiments for models and retrieval setups."""

    def __init__(self, config: AppConfig):
        self.config = config

    def benchmark_embeddings(
        self,
        sample_texts: list[str],
        models: list[EmbeddingModelConfig] | None = None,
    ) -> list[EmbeddingBenchmarkRow]:
        """Benchmark embedding model latency and memory."""
        model_configs = models or [self.config.embedding.primary, *self.config.embedding.comparisons]
        output: list[EmbeddingBenchmarkRow] = []

        for model_cfg in model_configs:
            backend = build_embedding_backend(model_cfg, self.config)
            tracemalloc.start()
            t0 = time.perf_counter()
            embeddings = embed_text_batches(sample_texts, backend=backend, batch_size=model_cfg.batch_size)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            row = EmbeddingBenchmarkRow(
                model_name=model_cfg.model_name,
                provider=model_cfg.provider,
                dimension=int(embeddings.shape[1]),
                avg_latency_ms=elapsed_ms / max(len(sample_texts), 1),
                max_memory_mb=peak / (1024 * 1024),
            )
            output.append(row)

        return output

    def benchmark_retrieval(
        self,
        system_name: str,
        evaluation_cases: list[EvaluationCase],
        search_fn,
        index_size_bytes: int,
        embedding_dim: int,
        chunk_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        retriever_mode: str,
        reranker_enabled: bool,
        embedding_model: str,
    ) -> BenchmarkRun:
        """Benchmark retrieval quality and latency."""
        eval_output: EvaluationOutput = evaluate_retrieval(
            system_name=system_name,
            cases=evaluation_cases,
            search_fn=search_fn,
            k=10,
        )
        latency_values = [float(row["latency_ms"]) for row in eval_output.per_query]
        p95_latency = float(np.percentile(latency_values, 95)) if latency_values else 0.0

        return BenchmarkRun(
            run_id=f"{system_name}-{int(time.time())}",
            embedding_model=embedding_model,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retriever_mode=retriever_mode,
            reranker_enabled=reranker_enabled,
            index_size_bytes=index_size_bytes,
            embedding_dim=embedding_dim,
            avg_query_latency_ms=eval_output.summary.avg_latency_ms,
            p95_query_latency_ms=p95_latency,
            precision_at_10=eval_output.summary.precision_at_k,
            recall_at_10=eval_output.summary.recall_at_k,
            mrr=eval_output.summary.mrr,
            ndcg_at_10=eval_output.summary.ndcg,
        )


def estimate_dir_size_bytes(path: str | Path) -> int:
    """Estimate total file size in directory."""
    total = 0
    for root, _, files in os.walk(path):
        for file in files:
            fp = Path(root) / file
            total += fp.stat().st_size
    return total


def summarize_benchmark_runs(runs: list[BenchmarkRun]) -> dict[str, float]:
    """Aggregate run statistics."""
    if not runs:
        return {
            "avg_ndcg_at_10": 0.0,
            "avg_mrr": 0.0,
            "avg_latency_ms": 0.0,
        }
    return {
        "avg_ndcg_at_10": mean([run.ndcg_at_10 for run in runs]),
        "avg_mrr": mean([run.mrr for run in runs]),
        "avg_latency_ms": mean([run.avg_query_latency_ms for run in runs]),
    }
