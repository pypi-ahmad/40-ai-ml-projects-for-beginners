#!/usr/bin/env python3
"""Run end-to-end verification and leakage audit for the hybrid forecasting project."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import build_horizon_target, load_stock_data, split_data
from src.features import FeaturePipeline
from src.forecast_pipeline import ForecastingFramework
from src.models import MODEL_REGISTRY


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def dataset_audit(df: pd.DataFrame) -> dict[str, Any]:
    missing_values = int(df.isna().sum().sum())
    duplicate_dates = int(df.index.duplicated().sum())
    duplicate_rows = int(df.duplicated().sum())
    missing_business_days = pd.date_range(df.index.min(), df.index.max(), freq="B").difference(df.index)

    audit = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "is_chronological": bool(df.index.is_monotonic_increasing),
        "missing_values_total": missing_values,
        "duplicate_dates": duplicate_dates,
        "duplicate_rows": duplicate_rows,
        "invalid_high_low_rows": int((df["High"] < df["Low"]).sum()),
        "invalid_ohlc_positive_rows": int((df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()),
        "invalid_negative_volume_rows": int((df["Volume"] < 0).sum()),
        "missing_business_days_count": int(len(missing_business_days)),
        "missing_business_days_sample": [str(d.date()) for d in missing_business_days[:10]],
    }
    return audit


def leakage_audit(framework: ForecastingFramework, horizon: int) -> dict[str, Any]:
    cfg = framework.config["features"]
    raw = framework.df_raw if framework.df_raw is not None else framework.load_data()
    raw = raw.copy()

    pipe = FeaturePipeline(
        lags=cfg.get("lags"),
        rolling_windows=cfg.get("rolling_windows"),
        ema_windows=cfg.get("ema_windows"),
        wma_windows=cfg.get("wma_windows"),
        momentum_windows=cfg.get("momentum_windows"),
        include_technical=cfg.get("include_technical", True),
        include_date_features=cfg.get("include_date_features", True),
        include_price_derived=cfg.get("include_price_derived", True),
        dropna=False,
    )

    features_a = pipe.fit_transform(raw)
    cut = int(len(raw) * 0.8)
    mutated = raw.copy()
    scale_map = {
        "Open": 1.05,
        "High": 1.07,
        "Low": 1.03,
        "Close": 1.06,
        "Volume": 1.25,
        "AdjustedClose": 1.06,
    }
    for col, scale in scale_map.items():
        if col in mutated.columns:
            mutated[col] = mutated[col].astype(float)
            mutated.loc[mutated.index[cut:], col] = mutated.loc[mutated.index[cut:], col] * scale
    features_b = pipe.fit_transform(mutated)

    compare_cols = [c for c in features_a.columns if c not in {"day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year", "is_month_start", "is_month_end"}]
    diff = (
        features_a.iloc[:cut][compare_cols].fillna(0.0).to_numpy()
        - features_b.iloc[:cut][compare_cols].fillna(0.0).to_numpy()
    )
    past_changed_cells = int(np.count_nonzero(np.abs(diff) > 1e-12))

    target_df = build_horizon_target(features_a, target_col=cfg.get("target_col", "Close"), horizon=horizon).dropna()
    idx = min(100, len(target_df) - 1)
    row_index = target_df.index[idx]
    base_pos = features_a.index.get_loc(row_index)
    compare_pos = min(base_pos + horizon, len(features_a) - 1)
    target_alignment_error = float(
        abs(
            target_df.iloc[idx]["target"]
            - features_a.iloc[compare_pos][cfg.get("target_col", "Close")]
        )
    )

    train, val, test = split_data(
        target_df,
        train_end=framework.config["data"].get("train_end"),
        val_end=framework.config["data"].get("val_end"),
    )
    split_ok = bool(train.index.max() < val.index.min() and val.index.max() < test.index.min())

    return {
        "horizon": horizon,
        "past_feature_changes_after_future_mutation": past_changed_cells,
        "target_alignment_abs_error": target_alignment_error,
        "chronological_split_non_overlap": split_ok,
        "scaler_train_only_note": "Deep-learning scalers are fit on train and only transformed on val/test (see src/deep_learning.py).",
    }


def execution_audit(framework: ForecastingFramework, horizon: int) -> dict[str, Any]:
    baseline = framework.train_baselines(horizon)
    deep = framework.train_deep_models(horizon)
    hybrid = framework.train_hybrids(horizon, baseline_bundle=baseline, deep_bundle=deep)
    weight = framework.optimize_weights(
        horizon=horizon,
        predictions=hybrid["val_predictions"],
        y_true=hybrid["y_val_true"],
        method="grid",
        evaluation_predictions=hybrid["test_predictions"],
        evaluation_y_true=hybrid["y_test_true"],
    )
    backtest = framework.backtest(horizon=horizon, model=MODEL_REGISTRY["Random Forest"], strategy="walk_forward")

    return {
        "horizon": horizon,
        "baseline_best_model": str(baseline["leaderboard"].iloc[0]["model"]),
        "baseline_best_test_rmse": float(baseline["leaderboard"].iloc[0]["test_rmse"]),
        "deep_best_model": str(deep["leaderboard"].iloc[0]["model"]),
        "deep_best_test_rmse": float(deep["leaderboard"].iloc[0]["test_rmse"]),
        "hybrid_best_model": str(hybrid["leaderboard"].iloc[0]["model"]),
        "hybrid_best_test_rmse": float(hybrid["leaderboard"].iloc[0]["rmse"]),
        "optimized_weight_test_rmse": float(weight["test_metrics"]["rmse"]),
        "backtest_walk_forward_mean_rmse": float(backtest["aggregated_metrics"]["mean_rmse"]),
    }


def run(config_path: str, horizon: int) -> dict[str, Any]:
    framework = ForecastingFramework(config_path=config_path)
    df = framework.load_data()

    result = {
        "dataset_audit": dataset_audit(df),
        "leakage_audit": leakage_audit(framework, horizon=horizon),
        "execution_audit": execution_audit(framework, horizon=horizon),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid ML forecasting verification audit")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--output-json",
        default="outputs/artifacts/verification_summary.json",
        help="Where to save JSON audit output.",
    )
    args = parser.parse_args()

    summary = run(config_path=args.config, horizon=args.horizon)
    output_path = (PROJECT_ROOT / args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Verification audit written to %s", output_path)
    logger.info("Execution summary: %s", summary["execution_audit"])


if __name__ == "__main__":
    main()
