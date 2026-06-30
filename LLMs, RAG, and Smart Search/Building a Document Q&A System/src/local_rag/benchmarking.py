"""Performance benchmarking utilities for ingestion/retrieval/generation pipeline."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from local_rag.app import AppRuntime
from local_rag.performance import capture_resource_snapshot
from local_rag.retriever import RetrievalStrategy
from local_rag.utils import json_dump


@dataclass(slots=True)
class BenchmarkResult:
    """Aggregated benchmark snapshot."""

    retrieval_strategy: RetrievalStrategy
    query_count: int
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    avg_total_latency_ms: float
    queries_per_second: float
    vectors: int
    memory_mb: float
    disk_usage_mb: float


class BenchmarkRunner:
    """Run repeatable benchmarks on live runtime pipeline."""

    def __init__(self, runtime: AppRuntime) -> None:
        self.runtime = runtime

    def run_queries(
        self,
        queries: list[str],
        *,
        top_k: int,
        strategy: RetrievalStrategy,
    ) -> BenchmarkResult:
        """Benchmark repeated queries for given retrieval strategy."""

        rows = [query.strip() for query in queries if query.strip()]
        if not rows:
            raise ValueError("No benchmark queries provided.")

        retrieval_ms: list[float] = []
        generation_ms: list[float] = []
        total_ms: list[float] = []

        started = time.perf_counter()
        for query in rows:
            response = self.runtime.pipeline.ask(query, top_k=top_k, strategy=strategy)
            retrieval_ms.append(response.timings.retrieval_ms)
            generation_ms.append(response.timings.generation_ms)
            total_ms.append(response.timings.total_ms)
        elapsed = max(time.perf_counter() - started, 1e-9)

        snapshot = capture_resource_snapshot(self.runtime.settings.vector_db_path)
        return BenchmarkResult(
            retrieval_strategy=strategy,
            query_count=len(rows),
            avg_retrieval_latency_ms=statistics.mean(retrieval_ms),
            avg_generation_latency_ms=statistics.mean(generation_ms),
            avg_total_latency_ms=statistics.mean(total_ms),
            queries_per_second=len(rows) / elapsed,
            vectors=self.runtime.vector_store.count(),
            memory_mb=snapshot.rss_mb,
            disk_usage_mb=snapshot.disk_usage_mb,
        )

    @staticmethod
    def save(path: Path, rows: list[BenchmarkResult]) -> None:
        """Persist benchmark rows to JSON report."""

        payload = {"rows": [asdict(row) for row in rows], "count": len(rows)}
        json_dump(path, payload)
