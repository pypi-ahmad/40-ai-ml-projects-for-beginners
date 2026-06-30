"""Benchmark report rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from domain_llm_ft.benchmark.runner import BenchmarkResult


def benchmark_table(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to DataFrame."""
    return pd.DataFrame(
        {
            "model": [r.model for r in results],
            "latency_ms": [r.latency_ms for r in results],
            "throughput": [r.throughput for r in results],
            "memory_mb": [r.memory_mb for r in results],
        }
    )


def render_benchmark_charts(table: pd.DataFrame, output_dir: Path) -> None:
    """Render benchmark summary charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.bar(table["model"], table["latency_ms"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Latency (ms/sample)")
    plt.tight_layout()
    plt.savefig(output_dir / "latency.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(table["model"], table["throughput"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Throughput (samples/sec)")
    plt.tight_layout()
    plt.savefig(output_dir / "throughput.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(table["model"], table["memory_mb"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Peak Memory (MB)")
    plt.tight_layout()
    plt.savefig(output_dir / "memory.png")
    plt.close()
