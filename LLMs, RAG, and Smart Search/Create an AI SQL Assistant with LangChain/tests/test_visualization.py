from __future__ import annotations

from ai_sql_assistant.visualization.recommender import recommend_visualizations


def test_visualization_recommendations_include_table() -> None:
    rows = [
        {"month": "2024-01", "revenue": 1000.0},
        {"month": "2024-02", "revenue": 1200.0},
    ]
    specs = recommend_visualizations(rows)
    chart_types = [item.chart_type for item in specs]

    assert "table" in chart_types
    assert "line" in chart_types or "time_series" in chart_types
