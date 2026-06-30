"""Run benchmark matrix and persist metrics/figures.

Usage:
    UV_CACHE_DIR=.uv_cache uv run python scripts/run_benchmark.py --runs 3 --prompt-mode medium
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.config import APP_CONFIG
from streamlit_app.utils.helpers import FIGURES_DIR, METRICS_DIR, load_sample_text
from streamlit_app.utils.models import is_ollama_available, run_benchmark_matrix


PROMPTS = {
    "short": load_sample_text("sentiment_neutral"),
    "medium": load_sample_text("summary"),
    "long": "\n\n".join([load_sample_text("summary")] * 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Ollama models")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per model")
    parser.add_argument("--prompt-mode", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def save_figures(summary_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    latency_fig = px.bar(summary_df, x="model", y="mean_latency", color="model", title="Mean latency")
    latency_fig.write_html(str(FIGURES_DIR / "cli_benchmark_latency.html"), include_plotlyjs="cdn")

    throughput_fig = px.bar(
        summary_df,
        x="model",
        y="mean_throughput_wps",
        color="model",
        title="Mean throughput (words/sec)",
    )
    throughput_fig.write_html(str(FIGURES_DIR / "cli_benchmark_throughput.html"), include_plotlyjs="cdn")


def main() -> int:
    args = parse_args()
    if not is_ollama_available():
        print("ERROR: Ollama daemon unavailable. Start with `ollama serve`.")
        return 1

    prompt = PROMPTS[args.prompt_mode]
    summaries, run_rows = run_benchmark_matrix(
        models=APP_CONFIG.models.benchmark_models,
        prompt=prompt,
        runs=args.runs,
        system_prompt=None,
        temperature=args.temperature,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)
    runs_df = pd.DataFrame(run_rows)

    summary_csv = METRICS_DIR / f"cli_benchmark_summary_{timestamp}.csv"
    runs_csv = METRICS_DIR / f"cli_benchmark_runs_{timestamp}.csv"
    summary_json = METRICS_DIR / f"cli_benchmark_summary_{timestamp}.json"

    summary_df.to_csv(summary_csv, index=False)
    runs_df.to_csv(runs_csv, index=False)
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    save_figures(summary_df)

    print(f"Saved: {summary_csv}")
    print(f"Saved: {runs_csv}")
    print(f"Saved: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
