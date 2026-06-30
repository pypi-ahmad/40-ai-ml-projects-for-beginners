"""Run benchmark suite and persist metrics/figures for docs and notebooks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.benchmarking import BENCHMARK_MODELS, BenchmarkRunner, format_benchmark_table
from src.visualization import BenchmarkVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark script."""

    parser = argparse.ArgumentParser(description="Run local Ollama benchmarks.")
    parser.add_argument("--runs", type=int, default=3, help="Inference runs per model.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=BENCHMARK_MODELS,
        help="Model tags to benchmark.",
    )
    parser.add_argument(
        "--primary-prompt",
        choices=["short", "medium", "long"],
        default="medium",
        help="Prompt group used for top-level benchmark_results.* outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute benchmark suite and save outputs to outputs/ directory."""

    args = parse_args()
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner()
    visualizer = BenchmarkVisualizer()

    try:
        short_results = runner.run_all(prompt_key="short", runs=args.runs, models=args.models)
        medium_results = runner.run_all(prompt_key="medium", runs=args.runs, models=args.models)
        long_results = runner.run_all(prompt_key="long", runs=args.runs, models=args.models)

        results_by_prompt = {
            "short": short_results,
            "medium": medium_results,
            "long": long_results,
        }

        manifest = runner.export_bundle(
            results_by_prompt=results_by_prompt,
            output_dir=str(output_dir),
            primary_prompt=args.primary_prompt,
        )

        figures = visualizer.generate_all(
            medium_results=medium_results,
            short_results=short_results,
            long_results=long_results,
        )

        primary_results = results_by_prompt[args.primary_prompt]
        markdown_table = format_benchmark_table(primary_results)

        benchmark_table_path = Path(manifest["benchmark_table"])
        benchmark_table_path.write_text(markdown_table, encoding="utf-8")

        summary = {
            "manifest": manifest,
            "figures": figures,
        }
        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info("Benchmark run complete. Artifacts saved to %s", output_dir)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
