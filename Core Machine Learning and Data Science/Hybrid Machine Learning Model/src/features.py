from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["Open", "High", "Low", "Close"]



def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")



def weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)



def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, PRICE_COLUMNS)
    out = df.copy()
    out["daily_return"] = out["Close"].pct_change()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["pct_change"] = out["Close"].pct_change() * 100.0
    out["intraday_range"] = out["High"] - out["Low"]
    out["intraday_range_pct"] = (out["High"] - out["Low"]) / out["Open"].replace(0, np.nan)
    return out



def add_trend_features(
    df: pd.DataFrame,
    sma_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    wma_windows: list[int] | None = None,
) -> pd.DataFrame:
    _require_columns(df, ["Close"])
    out = df.copy()
    sma_windows = sma_windows or [5, 10, 20, 50]
    ema_windows = ema_windows or [5, 10, 20, 50]
    wma_windows = wma_windows or [5, 10, 20]

    for w in sma_windows:
        out[f"sma_{w}"] = out["Close"].rolling(w).mean()
    for w in ema_windows:
        out[f"ema_{w}"] = out["Close"].ewm(span=w, adjust=False).mean()
    for w in wma_windows:
        out[f"wma_{w}"] = weighted_moving_average(out["Close"], w)

    return out



def add_momentum_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    _require_columns(df, ["Close"])
    out = df.copy()
    windows = windows or [5, 10, 20]

    for w in windows:
        out[f"roc_{w}"] = out["Close"].pct_change(w) * 100.0
        out[f"momentum_{w}"] = out["Close"] - out["Close"].shift(w)

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out["rsi_14"] = 100 - (100 / (1 + rs))

    return out



def add_volatility_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    _require_columns(df, PRICE_COLUMNS)
    out = df.copy()
    windows = windows or [5, 10, 20, 50]

    for w in windows:
        out[f"rolling_std_{w}"] = out["daily_return"].rolling(w).std()

    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift(1)).abs()
    low_close = (out["Low"] - out["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean()

    basis = out["Close"].rolling(20).mean()
    dev = out["Close"].rolling(20).std()
    out["bb_mid_20"] = basis
    out["bb_upper_20"] = basis + 2 * dev
    out["bb_lower_20"] = basis - 2 * dev
    out["bb_width_20"] = out["bb_upper_20"] - out["bb_lower_20"]

    return out



def add_volume_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    _require_columns(df, ["Volume"])
    out = df.copy()
    windows = windows or [5, 10, 20]

    out["volume_change"] = out["Volume"].pct_change()
    for w in windows:
        out[f"volume_ma_{w}"] = out["Volume"].rolling(w).mean()
    out["volume_z_20"] = (
        (out["Volume"] - out["Volume"].rolling(20).mean())
        / out["Volume"].rolling(20).std().replace(0, np.nan)
    )
    return out



def add_lagged_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lags: list[int] | None = None,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add lag features.

    `cols` kept for backward compatibility with older tests.
    """
    out = df.copy()
    lags = lags or [1, 3, 5, 10, 20, 60]
    selected = columns or cols or ["Close"]

    if any(lag <= 0 for lag in lags):
        raise ValueError("lags must be positive integers")

    for col in selected:
        if col not in out.columns:
            raise ValueError(f"Column '{col}' not present for lag features")
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out



def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        return out

    out["day_of_week"] = out.index.dayofweek
    out["day_of_month"] = out.index.day
    out["week_of_year"] = out.index.isocalendar().week.astype(int)
    out["month"] = out.index.month
    out["quarter"] = out.index.quarter
    out["year"] = out.index.year
    out["is_month_start"] = out.index.is_month_start.astype(int)
    out["is_month_end"] = out.index.is_month_end.astype(int)
    return out



def add_price_position_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, PRICE_COLUMNS + ["Volume"])
    out = df.copy()
    out["price_mid"] = (out["High"] + out["Low"]) / 2.0
    out["close_position"] = (
        (out["Close"] - out["Low"])
        / (out["High"] - out["Low"]).replace(0, np.nan)
    )
    out["vwap_20"] = (
        (out["Close"] * out["Volume"]).rolling(20).sum()
        / out["Volume"].rolling(20).sum().replace(0, np.nan)
    )
    out["close_to_vwap"] = out["Close"] / out["vwap_20"] - 1.0
    return out



def create_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    wma_windows: list[int] | None = None,
    momentum_windows: list[int] | None = None,
    include_technical: bool = True,
    include_date_features: bool = True,
    include_price_derived: bool = True,
    dropna: bool = False,
) -> pd.DataFrame:
    """Create production-ready financial feature set."""
    out = df.copy()
    out = add_price_features(out)
    out = add_trend_features(
        out,
        sma_windows=rolling_windows,
        ema_windows=ema_windows,
        wma_windows=wma_windows,
    )
    if include_technical:
        out = add_momentum_features(out, windows=momentum_windows)
        out = add_volatility_features(out, windows=rolling_windows)
        out = add_volume_features(out, windows=rolling_windows[:3] if rolling_windows else None)
    out = add_lagged_features(out, columns=["Close", "Volume"], lags=lags)
    if include_price_derived:
        out = add_price_position_features(out)
    if include_date_features:
        out = add_date_features(out)

    if dropna:
        out = out.dropna()
    return out


@dataclass(slots=True)
class FeaturePipeline:
    lags: list[int] | None = None
    rolling_windows: list[int] | None = None
    ema_windows: list[int] | None = None
    wma_windows: list[int] | None = None
    momentum_windows: list[int] | None = None
    include_technical: bool = True
    include_date_features: bool = True
    include_price_derived: bool = True
    dropna: bool = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        # Stateless transform pipeline, kept for sklearn-like API.
        _ = create_features(
            df,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            ema_windows=self.ema_windows,
            wma_windows=self.wma_windows,
            momentum_windows=self.momentum_windows,
            include_technical=self.include_technical,
            include_date_features=self.include_date_features,
            include_price_derived=self.include_price_derived,
            dropna=False,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_features(
            df,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            ema_windows=self.ema_windows,
            wma_windows=self.wma_windows,
            momentum_windows=self.momentum_windows,
            include_technical=self.include_technical,
            include_date_features=self.include_date_features,
            include_price_derived=self.include_price_derived,
            dropna=self.dropna,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
