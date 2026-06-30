from __future__ import annotations

import numpy as np
import pandas as pd

import modules.feature_selector as fs
from modules.settings import load_config


def test_feature_selection_uses_temporal_split(monkeypatch) -> None:
    """Ensure feature ranking is learned from train partition only."""
    config = load_config()
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=20, freq="D"),
            "App": ["YouTube"] * 20,
            "Usage (minutes)": np.linspace(10, 30, 20),
            "Notifications": np.random.randint(1, 10, size=20),
            "Times Opened": np.random.randint(1, 15, size=20),
            "feature_a": np.random.randn(20),
            "feature_b": np.random.randn(20),
            "target_next_day": np.linspace(11, 31, 20),
        }
    )

    called = {"value": False}

    def fake_split(df: pd.DataFrame, date_col: str, test_size: float):
        called["value"] = True
        split_at = int(len(df) * (1 - test_size))
        return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()

    monkeypatch.setattr(fs, "temporal_train_test_split", fake_split)
    ranking = fs.run_feature_selection(frame, config=config)

    assert called["value"] is True
    assert not ranking.empty
