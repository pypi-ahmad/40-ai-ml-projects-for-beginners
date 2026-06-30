from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd


REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Volume"}
CLOSE_CANDIDATES = ["Close", "Close/Last", "Adj Close", "Adjusted Close"]


@dataclass(slots=True)
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame



def _clean_price_column(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )



def load_stock_data(path: str | Path) -> pd.DataFrame:
    """Load and validate OHLCV stock data.

    Args:
        path: CSV path containing Date and OHLCV columns.

    Returns:
        Chronologically sorted DataFrame with DatetimeIndex.

    Example:
        >>> df = load_stock_data("data/apple_stock_data.csv")
        >>> list(df.columns)
        ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    missing_base = REQUIRED_COLUMNS.difference(df.columns)
    if missing_base:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_base)}")

    close_col = next((col for col in CLOSE_CANDIDATES if col in df.columns), None)
    if close_col is None:
        raise ValueError(
            f"Dataset missing close column. Expected one of: {CLOSE_CANDIDATES}"
        )

    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjustedClose"})
    if "Adjusted Close" in df.columns:
        df = df.rename(columns={"Adjusted Close": "AdjustedClose"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Date column contains unparseable values")
    if df["Date"].duplicated().any():
        raise ValueError("Date column contains duplicate timestamps")
    if df.duplicated().any():
        raise ValueError("Dataset contains duplicate rows")

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = _clean_price_column(df[col])

    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    if df[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
        raise ValueError("Numeric OHLCV columns contain invalid values")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC values must be strictly positive")
    if (df["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if (df["High"] < df["Low"]).any():
        raise ValueError("Found rows where High < Low")
    if ((df["Open"] < df["Low"]) | (df["Open"] > df["High"])).any():
        warnings.warn("Open price falls outside High/Low range for some rows", UserWarning)
    if ((df["Close"] < df["Low"]) | (df["Close"] > df["High"])).any():
        warnings.warn("Close price falls outside High/Low range for some rows", UserWarning)

    df = df.set_index("Date").sort_index()
    if not df.index.is_monotonic_increasing:
        warnings.warn("Datetime index is not sorted. Sorting chronologically.", UserWarning)
        df = df.sort_index()

    return df



def split_data(
    df: pd.DataFrame,
    train_end: str | None = "2018-12-31",
    val_end: str | None = "2019-12-31",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split with date or ratio strategy."""
    if df.empty:
        raise ValueError("Input dataframe is empty")
    if len(df) < 3:
        raise ValueError("Need at least 3 rows for train/val/test split")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        warnings.warn("Input index not sorted. Sorting before split.", UserWarning)
        df = df.sort_index()

    if train_end is not None:
        train_cut = pd.Timestamp(train_end)
        if train_cut <= df.index.min() or train_cut >= df.index.max():
            raise ValueError("train_end must be after dataset start and before dataset end")

        if val_end is None:
            remainder = df.loc[df.index > train_cut]
            if len(remainder) < 2:
                raise ValueError("Not enough rows after train_end for val/test split")
            mid = len(remainder) // 2
            train = df.loc[df.index <= train_cut]
            val = remainder.iloc[:mid]
            test = remainder.iloc[mid:]
        else:
            val_cut = pd.Timestamp(val_end)
            if val_cut <= train_cut:
                raise ValueError("val_end must be after train_end")
            if val_cut >= df.index.max():
                raise ValueError("val_end must be before dataset end")

            train = df.loc[df.index <= train_cut]
            val = df.loc[(df.index > train_cut) & (df.index <= val_cut)]
            test = df.loc[df.index > val_cut]
    else:
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        if train_size <= 0 or val_size <= 0 or train_size + val_size >= n:
            raise ValueError("Invalid ratio split parameters")

        train = df.iloc[:train_size]
        val = df.iloc[train_size : train_size + val_size]
        test = df.iloc[train_size + val_size :]

    if train.empty or val.empty or test.empty:
        raise ValueError("Split produced empty partition(s)")

    return train.copy(), val.copy(), test.copy()



def build_horizon_target(
    df: pd.DataFrame,
    target_col: str = "Close",
    horizon: int = 1,
    target_name: str = "target",
) -> pd.DataFrame:
    """Create horizon target column using forward shift.

    Args:
        df: Feature dataframe.
        target_col: Base column to forecast.
        horizon: Forecast horizon in trading days.
        target_name: Name of generated target column.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found")

    out = df.copy()
    out[target_name] = out[target_col].shift(-horizon)
    return out
