"""Benchmark table generation utilities."""

from __future__ import annotations

import csv
from pathlib import Path


def write_benchmark_tables(results: list[dict[str, float | str | int]], output_dir: str | Path) -> tuple[Path, Path]:
    """Write benchmark results to CSV and Markdown table."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "latency_benchmark.csv"
    md_path = out_dir / "latency_benchmark.md"

    headers = ["backend", "prompt_count", "mean_latency_ms"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow({header: row.get(header, "") for header in headers})

    lines = [
        "| Backend | Prompt Count | Mean Latency (ms) |",
        "|---|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.get('backend', '')} | {row.get('prompt_count', '')} | {row.get('mean_latency_ms', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, md_path
