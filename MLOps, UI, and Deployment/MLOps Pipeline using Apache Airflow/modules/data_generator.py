"""Synthetic data augmentation for robust local time-series training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import load_dataset, save_dataset
from .settings import load_config


@dataclass
class SyntheticStats:
    """Profile stats for one app."""

    usage_mean: float
    usage_std: float
    notif_mean: float
    notif_std: float
    opened_mean: float
    opened_std: float


def _profile_by_app(df: pd.DataFrame, app_col: str, target_col: str) -> dict[str, SyntheticStats]:
    profiles: dict[str, SyntheticStats] = {}
    for app, gdf in df.groupby(app_col):
        profiles[str(app)] = SyntheticStats(
            usage_mean=float(gdf[target_col].mean()),
            usage_std=float(max(gdf[target_col].std(ddof=1), 3.0)),
            notif_mean=float(gdf["Notifications"].mean()),
            notif_std=float(max(gdf["Notifications"].std(ddof=1), 2.0)),
            opened_mean=float(gdf["Times Opened"].mean()),
            opened_std=float(max(gdf["Times Opened"].std(ddof=1), 1.5)),
        )
    return profiles


def generate_synthetic_data(
    raw_df: pd.DataFrame,
    days_to_generate: int,
    noise_scale: float,
    seasonal_scale: float,
    random_state: int,
    target_col: str = "Usage (minutes)",
    date_col: str = "Date",
    app_col: str = "App",
) -> pd.DataFrame:
    """Generate synthetic rows per app/day using profile statistics.

    Args:
        raw_df: Source dataframe containing historical data.
        days_to_generate: Number of future days to generate per app.
        noise_scale: Noise multiplier for Gaussian perturbation.
        seasonal_scale: Weekly seasonal multiplier amplitude.
        random_state: Seed for deterministic generation.
    """
    if raw_df.empty:
        raise ValueError("raw_df is empty")

    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, app_col, target_col, "Notifications", "Times Opened"])  # type: ignore[arg-type]

    profiles = _profile_by_app(df, app_col=app_col, target_col=target_col)
    rng = np.random.default_rng(random_state)

    start_date = df[date_col].max() + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=days_to_generate, freq="D")

    rows: list[dict[str, Any]] = []
    for app, stats in profiles.items():
        for current_date in dates:
            day_of_week = current_date.dayofweek
            weekend_boost = 1.0 + (0.22 if day_of_week >= 5 else 0.0)
            seasonal_wave = 1.0 + seasonal_scale * np.sin(2 * np.pi * day_of_week / 7.0)
            factor = weekend_boost * seasonal_wave

            usage = stats.usage_mean * factor + rng.normal(0, stats.usage_std * noise_scale)
            notifications = stats.notif_mean * factor + rng.normal(0, stats.notif_std * noise_scale)
            times_opened = stats.opened_mean * factor + rng.normal(0, stats.opened_std * noise_scale)

            rows.append(
                {
                    date_col: current_date,
                    app_col: app,
                    target_col: round(float(max(usage, 1.0)), 2),
                    "Notifications": int(max(round(notifications), 0)),
                    "Times Opened": int(max(round(times_opened), 1)),
                    "Synthetic": 1,
                }
            )

    synthetic_df = pd.DataFrame(rows)

    base = df[[date_col, app_col, target_col, "Notifications", "Times Opened"]].copy()
    base["Synthetic"] = 0
    combined = pd.concat([base, synthetic_df], ignore_index=True)
    combined = combined.sort_values([date_col, app_col]).reset_index(drop=True)
    return combined


def run_data_augmentation(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load raw data, build synthetic augmentation, persist to configured path."""
    config = config or load_config()

    raw_df = load_dataset("raw", config=config)
    cfg = config["synthetic_data"]
    project = config["project"]

    augmented_df = generate_synthetic_data(
        raw_df=raw_df,
        days_to_generate=int(cfg["days_to_generate"]),
        noise_scale=float(cfg["noise_scale"]),
        seasonal_scale=float(cfg["seasonal_scale"]),
        random_state=int(project["random_state"]),
        target_col=str(project["target_col"]),
        date_col=str(project["date_col"]),
        app_col=str(project["group_col"]),
    )

    save_dataset(augmented_df, "synthetic", config=config)
    return augmented_df
