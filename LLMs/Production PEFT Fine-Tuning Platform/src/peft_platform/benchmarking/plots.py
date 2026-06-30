"""Benchmark plotting helpers."""

from __future__ import annotations

from pathlib import Path

from peft_platform.benchmarking.suite import BenchmarkResult
from peft_platform.utils.io import ensure_dir


def save_latency_bar(results: dict[str, BenchmarkResult], output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    labels = list(results.keys())
    values = [results[label].latency_ms_avg for label in labels]

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Average Latency by Method")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path
    except Exception:
        fallback = output_path.with_suffix(".txt")
        fallback.write_text(
            "\n".join(f"{label}: {value:.4f}" for label, value in zip(labels, values, strict=True)),
            encoding="utf-8",
        )
        return fallback
