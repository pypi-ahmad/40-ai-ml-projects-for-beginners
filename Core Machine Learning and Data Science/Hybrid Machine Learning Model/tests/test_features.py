from __future__ import annotations

import pandas as pd

from src.data_loader import load_stock_data
from src.features import FeaturePipeline, add_lagged_features, create_features



def test_add_lagged_features_columns(project_root):
    df = load_stock_data(project_root / "data" / "apple_stock_data.csv").head(200)
    out = add_lagged_features(df, columns=["Close"], lags=[1, 3, 5])
    assert "Close_lag_1" in out.columns
    assert "Close_lag_3" in out.columns
    assert "Close_lag_5" in out.columns



def test_create_features_adds_expected_signals(project_root):
    df = load_stock_data(project_root / "data" / "apple_stock_data.csv").head(500)
    feat = create_features(df, lags=[1, 3, 5], rolling_windows=[5, 10], dropna=False)
    expected = {"daily_return", "log_return", "sma_5", "ema_5", "rsi_14", "atr_14", "Close_lag_5"}
    assert expected.issubset(set(feat.columns))



def test_feature_pipeline_transform(project_root):
    df = load_stock_data(project_root / "data" / "apple_stock_data.csv").head(400)
    pipe = FeaturePipeline(lags=[1, 3], rolling_windows=[5, 10], dropna=True)
    out = pipe.fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) < len(df)  # dropna removes warmup rows
    assert "Close_lag_3" in out.columns
