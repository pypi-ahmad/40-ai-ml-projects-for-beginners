#!/usr/bin/env python3
"""Run full hybrid forecasting workflow across configured horizons."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast_pipeline import ForecastingFramework


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run full hybrid forecasting pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--horizon", type=int, default=None, help="Optional single horizon override")
    args = parser.parse_args()

    framework = ForecastingFramework(config_path=args.config)
    framework.load_data()

    if args.horizon is not None:
        logger.info("Running single horizon %d", args.horizon)
        framework.run_horizon(args.horizon)
    else:
        logger.info("Running configured horizons")
        framework.run_all_horizons()

    logger.info("Training pipeline complete. Outputs in %s", Path("outputs").resolve())


if __name__ == "__main__":
    main()
