#!/usr/bin/env python3
"""Run backtests for configured horizons and strategies."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast_pipeline import ForecastingFramework
from src.models import MODEL_REGISTRY


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest forecasting models")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--strategy", default="walk_forward", choices=["walk_forward", "expanding", "rolling"])
    parser.add_argument("--model", default="Random Forest")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODEL_REGISTRY.keys())}")

    framework = ForecastingFramework(config_path=args.config)
    framework.load_data()

    result = framework.backtest(
        horizon=args.horizon,
        model=MODEL_REGISTRY[args.model],
        strategy=args.strategy,
    )

    logger.info("Backtest strategy=%s horizon=%d model=%s", args.strategy, args.horizon, args.model)
    logger.info("Aggregated metrics: %s", result["aggregated_metrics"])


if __name__ == "__main__":
    main()
