from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import build_horizon_target, load_stock_data, split_data



def test_load_stock_data_success(project_root):
    df = load_stock_data(project_root / "data" / "apple_stock_data.csv")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)
    assert df.index.is_monotonic_increasing



def test_load_stock_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stock_data(tmp_path / "missing.csv")



def test_split_data_date_boundaries(project_root):
    df = load_stock_data(project_root / "data" / "apple_stock_data.csv")
    train, val, test = split_data(df, train_end="2018-12-31", val_end="2019-12-31")
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert train.index.max() <= pd.Timestamp("2018-12-31")
    assert val.index.min() > pd.Timestamp("2018-12-31")
    assert val.index.max() <= pd.Timestamp("2019-12-31")
    assert test.index.min() > pd.Timestamp("2019-12-31")



def test_split_data_ratio_mode(project_root):
    df = load_stock_data(project_root / "data" / "apple_stock_data.csv")
    train, val, test = split_data(df, train_end=None, val_end=None, train_ratio=0.7, val_ratio=0.15)
    assert abs(len(train) / len(df) - 0.7) < 0.03
    assert abs(len(val) / len(df) - 0.15) < 0.03
    assert abs(len(test) / len(df) - 0.15) < 0.03



def test_build_horizon_target(project_root):
    df = load_stock_data(project_root / "data" / "apple_stock_data.csv")
    out = build_horizon_target(df, target_col="Close", horizon=5, target_name="target")
    assert "target" in out.columns
    assert out["target"].isna().sum() == 5


def test_load_adj_close_alias(tmp_path):
    csv_path = tmp_path / "adj_close_only.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Adj Close,Volume",
                "2020-01-02,100,101,99,100.5,1000000",
                "2020-01-03,101,102,100,101.3,1100000",
                "2020-01-06,102,103,101,102.1,1200000",
            ]
        ),
        encoding="utf-8",
    )
    df = load_stock_data(csv_path)
    assert "Close" in df.columns
    assert df["Close"].iloc[0] == pytest.approx(100.5)
