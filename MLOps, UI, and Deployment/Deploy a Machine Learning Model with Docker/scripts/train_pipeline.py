"""CLI entrypoint for model training, benchmarking, and artifact generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.benchmarking import run_training_pipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for runtime profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "deep"],
        default="balanced",
        help="Runtime/quality profile. deep is intended for full notebook-grade benchmarking.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training pipeline and print compact JSON summary."""
    args = parse_args()
    artifacts = run_training_pipeline(profile=args.profile)
    print(
        json.dumps(
            {
                "best_model": artifacts.model_name,
                "profile": args.profile,
                "top3": artifacts.ranking.head(3).to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
