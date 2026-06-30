"""Preflight verification script for local project readiness checks."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from src.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_command(command: list[str]) -> None:
    """Run command and raise on failure."""

    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for preflight checks."""

    parser = argparse.ArgumentParser(description="Run preflight quality gate for project.")
    parser.add_argument(
        "--profile",
        choices=["fast", "full"],
        default="fast",
        help="fast = quick local confidence, full = end-to-end deep validation.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=None,
        help="Inference runs per model for benchmark stage (full profile only).",
    )
    parser.add_argument("--skip-lint", action="store_true", help="Skip static lint checks.")
    parser.add_argument(
        "--skip-notebooks", action="store_true", help="Skip notebook execution (full profile)."
    )
    parser.add_argument(
        "--skip-benchmarks", action="store_true", help="Skip benchmark execution (full profile)."
    )
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="Skip Ollama runtime/model availability verification.",
    )
    return parser.parse_args()


def ensure_expected_artifacts() -> None:
    """Validate expected benchmark and figure artifacts exist."""

    expected_artifacts = [
        Path("outputs/benchmarks/benchmark_results.json"),
        Path("outputs/benchmarks/artifact_manifest.json"),
        Path("outputs/figures/latency_comparison.png"),
        Path("outputs/figures/ml_app_architecture.png"),
    ]

    missing = [artifact for artifact in expected_artifacts if not artifact.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Expected artifacts missing: {missing_list}")


def main() -> None:
    """Execute ordered preflight checks and artifact validations."""

    args = parse_args()
    cfg = get_config()

    if not args.skip_lint:
        run_command(["uv", "run", "ruff", "check", "app.py", "src", "scripts", "tests"])

    if not args.skip_runtime_check:
        run_command(["uv", "run", "python", "-m", "scripts.verify_runtime"])
    else:
        logger.warning("Skipping runtime check; Ollama/model readiness is not verified.")

    if args.profile == "fast":
        run_command(["uv", "run", "pytest", "-ra", "-m", "not integration"])
        logger.info("Fast preflight checks passed.")
        return

    run_command(["uv", "run", "pytest", "-ra"])

    benchmark_runs = args.benchmark_runs or cfg.benchmark_full_runs
    if not args.skip_benchmarks:
        run_command(
            [
                "uv",
                "run",
                "python",
                "-m",
                "scripts.run_benchmarks",
                "--runs",
                str(benchmark_runs),
            ]
        )

    if not args.skip_notebooks:
        run_command(["uv", "run", "python", "-m", "scripts.execute_notebooks"])

    ensure_expected_artifacts()
    logger.info("Full preflight checks passed. Core artifacts present.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Preflight aborted: command failed (exit=%s): %s",
            exc.returncode,
            " ".join(exc.cmd),
        )
        sys.exit(exc.returncode)
