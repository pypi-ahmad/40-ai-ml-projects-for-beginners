"""Feature engineering pipeline for mobile app usage forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_loader import save_dataset
from .settings import load_config


def _holiday_flag(series: pd.Series, country: str) -> pd.Series:
    """Return holiday indicator for given date series."""
    try:
        import holidays

        holiday_set = holidays.country_holidays(country)
        return series.dt.date.astype("object").map(lambda d: 1 if d in holiday_set else 0).astype(int)
    except Exception:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)


def create_temporal_features(df: pd.DataFrame, date_col: str, holiday_country: str) -> pd.DataFrame:
    """Create temporal calendar features."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    out["day"] = out[date_col].dt.day
    out["week"] = out[date_col].dt.isocalendar().week.astype(int)
    out["month"] = out[date_col].dt.month
    out["quarter"] = out[date_col].dt.quarter
    out["year"] = out[date_col].dt.year
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_month_start"] = out[date_col].dt.is_month_start.astype(int)
    out["is_month_end"] = out[date_col].dt.is_month_end.astype(int)
    out["is_holiday"] = _holiday_flag(out[date_col], country=holiday_country)
    return out


def create_lag_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    lag_columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Create lag features within each app group."""
    out = df.copy().sort_values([group_col, date_col])
    for col in lag_columns:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out.groupby(group_col)[col].shift(lag)
    return out


def create_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    rolling_columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Create trailing rolling mean/std features within each app group."""
    out = df.copy().sort_values([group_col, date_col])
    for col in rolling_columns:
        for window in windows:
            out[f"{col}_rolling_mean_{window}"] = out.groupby(group_col)[col].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
            out[f"{col}_rolling_std_{window}"] = out.groupby(group_col)[col].transform(
                lambda s: s.rolling(window=window, min_periods=2).std()
            )
    return out


def create_behavior_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create behavior and engagement features."""
    out = df.copy()
    out["notifications_per_usage_minute"] = out["Notifications"] / (out[target_col] + 1e-6)
    out["open_frequency"] = out["Times Opened"] / (out[target_col] + 1e-6)
    out["engagement_score"] = (
        0.5 * np.log1p(out[target_col])
        + 0.3 * np.log1p(out["Times Opened"])
        + 0.2 * np.log1p(out["Notifications"])
    )
    return out


def create_interaction_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create nonlinear interaction features."""
    out = df.copy()
    out["notifications_x_opens"] = out["Notifications"] * out["Times Opened"]
    out["usage_x_opens"] = out[target_col] * out["Times Opened"]
    return out


def create_forecast_target(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_col: str,
    horizon_days: int,
    output_col: str = "target_next_day",
) -> pd.DataFrame:
    """Shift target by horizon within each app to build forecasting label."""
    out = df.copy().sort_values([group_col, date_col])
    out[output_col] = out.groupby(group_col)[target_col].shift(-horizon_days)
    return out


def build_feature_pipeline(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Create full feature matrix and forecast target.

    Includes temporal, lag, rolling, behavior, interaction, and forecast target features.
    """
    config = config or load_config()
    project_cfg = config["project"]
    fe_cfg = config["feature_engineering"]

    target_col = str(project_cfg["target_col"])
    date_col = str(project_cfg["date_col"])
    group_col = str(project_cfg["group_col"])

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, group_col, target_col, "Notifications", "Times Opened"])

    if bool(fe_cfg.get("clip_negative_usage", True)):
        out[target_col] = out[target_col].clip(lower=0)

    out = create_temporal_features(
        out,
        date_col=date_col,
        holiday_country=str(fe_cfg["holiday_country"]),
    )
    out = create_lag_features(
        out,
        group_col=group_col,
        date_col=date_col,
        lag_columns=[target_col, "Notifications", "Times Opened"],
        lags=[int(x) for x in fe_cfg["lags"]],
    )
    out = create_rolling_features(
        out,
        group_col=group_col,
        date_col=date_col,
        rolling_columns=[target_col, "Notifications", "Times Opened"],
        windows=[int(x) for x in fe_cfg["rolling_windows"]],
    )
    out = create_behavior_features(out, target_col=target_col)
    out = create_interaction_features(out, target_col=target_col)
    out = create_forecast_target(
        out,
        group_col=group_col,
        date_col=date_col,
        target_col=target_col,
        horizon_days=int(project_cfg["forecast_horizon_days"]),
    )

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if bool(fe_cfg.get("drop_na_target", True)):
        out = out.dropna(subset=["target_next_day"])

    out = out.sort_values([date_col, group_col]).reset_index(drop=True)
    return out


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split."""
    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1 - test_size))
    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()
    return train_df, test_df


def run_feature_pipeline(input_df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run feature engineering and persist featured data parquet."""
    config = config or load_config()
    featured = build_feature_pipeline(input_df, config=config)
    save_dataset(featured, "features", config=config)
    return featured
