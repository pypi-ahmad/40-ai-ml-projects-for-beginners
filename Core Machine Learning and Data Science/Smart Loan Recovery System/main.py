#!/usr/bin/env python3
"""CLI entrypoint for Smart Loan Recovery System pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.loan_recovery import (
    DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    SmartLoanRecoveryPipeline,
    configure_logging,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Smart Loan Recovery end-to-end pipeline.")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to loan-recovery CSV file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--strict", action="store_true", help="Fail pipeline on blocking quality checks or tool failures.")
    return parser.parse_args()


def main() -> None:
    """Execute pipeline and print concise summary."""
    args = parse_args()
    configure_logging()

    pipeline = SmartLoanRecoveryPipeline(data_path=Path(args.data), random_state=args.seed, strict_mode=args.strict)
    artifacts = pipeline.run()

    print("=" * 80)
    print("SMART LOAN RECOVERY PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Best manual model: {artifacts.best_model_name}")
    print("\nClassification + business evaluation summary:")
    for section, values in artifacts.evaluation_summary.items():
        print(f"- {section}: {values}")

    print("\nOutput directories:")
    print(f"- Figures: {FIGURES_DIR}")
    print(f"- Tables: {TABLES_DIR}")
    print(f"- Reports: {REPORTS_DIR}")
    print(f"- Models: {MODELS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
