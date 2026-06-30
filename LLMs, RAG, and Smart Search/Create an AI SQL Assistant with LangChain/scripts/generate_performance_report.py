#!/usr/bin/env python3
"""Generate performance analysis markdown + plots from latest benchmark report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ai_sql_assistant.constants import PLOTS_DIR, REPORTS_DIR
from ai_sql_assistant.logging_utils import configure_logging, logger

matplotlib.use("Agg")


def latest_benchmark_report() -> Path:
    paths = sorted(REPORTS_DIR.glob("benchmark_run_*.json"))
    if not paths:
        raise FileNotFoundError("No benchmark_run_*.json found in artifacts/reports")
    return paths[-1]


def main() -> None:
    configure_logging()
    report_path = latest_benchmark_report()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    summary = pd.DataFrame(payload["metrics"]["summary"])
    if summary.empty:
        raise RuntimeError("Benchmark summary empty")

    summary["label"] = summary["approach"] + " | " + summary["model"]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generation latency plot.
    plt.figure(figsize=(10, 5))
    plt.bar(summary["label"], summary["avg_generation_latency_ms"], color="#4e79a7")
    plt.title("Generation Latency (ms)")
    plt.ylabel("ms")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    gen_plot = PLOTS_DIR / "benchmark_generation_latency.png"
    plt.savefig(gen_plot, dpi=160)
    plt.close()

    # Throughput plot.
    plt.figure(figsize=(10, 5))
    plt.bar(summary["label"], summary["query_throughput_qps"], color="#59a14f")
    plt.title("Query Throughput (QPS)")
    plt.ylabel("queries/second")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    throughput_plot = PLOTS_DIR / "benchmark_throughput.png"
    plt.savefig(throughput_plot, dpi=160)
    plt.close()

    judge_summary = payload.get("judge", {}).get("judge_summary", [])

    lines = [
        "# Performance Analysis",
        "",
        f"Source benchmark file: `{report_path.name}`",
        "",
        "## Core Metrics",
        "",
        "| Approach | Model | Gen Latency (ms) | Exec Latency (ms) | Rows | Tokens | Complexity | Memory (MB) | Throughput (QPS) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['approach']} | {row['model']} | {row['avg_generation_latency_ms']:.2f} | "
            f"{row['avg_execution_latency_ms']:.2f} | {row['avg_rows_returned']:.1f} | "
            f"{row['avg_token_count']:.1f} | {row['avg_complexity_score']:.2f} | "
            f"{row['avg_memory_mb']:.2f} | {row['query_throughput_qps']:.3f} |"
        )

    if judge_summary:
        lines.extend(
            [
                "",
                "## Judge Metrics",
                "",
                "| Approach | Model | SQL | Business | Completeness | Readability | Efficiency | Safety |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in judge_summary:
            lines.append(
                f"| {row['approach']} | {row['model']} | {row['sql_correctness']:.2f} | "
                f"{row['business_correctness']:.2f} | {row['completeness']:.2f} | "
                f"{row['readability']:.2f} | {row['efficiency']:.2f} | {row['safety']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Generated Plots",
            f"- `{gen_plot}`",
            f"- `{throughput_plot}`",
        ]
    )

    out = REPORTS_DIR / "performance_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Performance report: {}", out)
    logger.info("Generation latency plot: {}", gen_plot)
    logger.info("Throughput plot: {}", throughput_plot)


if __name__ == "__main__":
    main()
