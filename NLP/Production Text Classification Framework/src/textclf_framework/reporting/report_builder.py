"""Benchmark and evaluation report builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


class ReportBuilder:
    """Build markdown and HTML report artifacts from benchmark/evaluation data."""

    def __init__(self, report_dir: str | Path) -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def build_benchmark_summary(self, benchmark_csv: str | Path) -> Path:
        frame = pd.read_csv(benchmark_csv)
        out = self.report_dir / "benchmark_summary.md"

        top = frame.sort_values("macro_f1", ascending=False).head(10)
        lines = ["# Benchmark Summary", "", "## Top Configurations", "", top.to_markdown(index=False)]
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def build_benchmark_plot(self, benchmark_csv: str | Path) -> Path:
        frame = pd.read_csv(benchmark_csv)
        figure = px.bar(
            frame,
            x="model",
            y="macro_f1",
            color="dataset",
            barmode="group",
            title="Macro F1 by Model and Dataset",
        )
        output_path = self.report_dir / "benchmark_macro_f1.html"
        figure.write_html(output_path)
        return output_path
