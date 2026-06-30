from __future__ import annotations

import pandas as pd

from ai_spreadsheet_analytics.analytics import AnalyticsEngine


def test_run_full_eda_returns_core_sections() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "revenue": [100 + i * 5 for i in range(20)],
            "product": ["A", "B"] * 10,
        }
    )

    payload = AnalyticsEngine().run_full_eda(df)

    for key in [
        "summary",
        "correlations",
        "time_series",
        "kpis",
        "trend_detection",
        "forecast",
        "schema_inference",
    ]:
        assert key in payload
