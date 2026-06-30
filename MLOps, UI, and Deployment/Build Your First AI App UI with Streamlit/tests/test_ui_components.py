"""Smoke tests for UI components (mocked Streamlit)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

BASE_PATCH = "streamlit_app.components.ui_components.st"


@patch(f"{BASE_PATCH}.metric")
def test_render_metric_card(mock_metric) -> None:
    from streamlit_app.components.ui_components import render_metric_card

    render_metric_card("Latency", "123ms", "+2%")
    mock_metric.assert_called_once_with(label="Latency", value="123ms", delta="+2%")


@patch(f"{BASE_PATCH}.container", return_value=MagicMock())
@patch(f"{BASE_PATCH}.write")
@patch(f"{BASE_PATCH}.markdown")
def test_render_result_box(mock_markdown, mock_write, mock_container) -> None:
    from streamlit_app.components.ui_components import render_result_box

    render_result_box("content", "Title")
    mock_markdown.assert_called_once_with("**Title**")
    mock_write.assert_called_once_with("content")


@patch(f"{BASE_PATCH}.plotly_chart")
@patch(f"{BASE_PATCH}.dataframe")
@patch("streamlit_app.components.ui_components._save_plotly_figure")
def test_render_model_comparison(mock_save, mock_dataframe, mock_plotly_chart) -> None:
    from streamlit_app.components.ui_components import render_model_comparison

    summary = [
        {
            "model": "qwen3.5:2b",
            "mean_latency": 1.2,
            "mean_throughput_wps": 20.0,
            "mean_memory_mb": 1024.0,
            "mean_quality_score": 0.65,
        },
        {
            "model": "qwen3.5:4b",
            "mean_latency": 2.2,
            "mean_throughput_wps": 15.0,
            "mean_memory_mb": 1500.0,
            "mean_quality_score": 0.72,
        },
    ]

    fig = render_model_comparison(summary)
    assert fig is not None
    mock_dataframe.assert_called_once()
    mock_plotly_chart.assert_called()


@patch(f"{BASE_PATCH}.plotly_chart")
@patch("streamlit_app.components.ui_components._save_plotly_figure")
def test_render_latency_distribution(mock_save, mock_plotly_chart) -> None:
    from streamlit_app.components.ui_components import render_latency_distribution

    rows = [
        {"model": "qwen3.5:2b", "latency_seconds": 1.1},
        {"model": "qwen3.5:2b", "latency_seconds": 1.3},
    ]

    fig = render_latency_distribution(rows)
    assert fig is not None
    mock_plotly_chart.assert_called()


@patch(f"{BASE_PATCH}.plotly_chart")
def test_render_confidence_gauge(mock_plotly_chart) -> None:
    from streamlit_app.components.ui_components import render_confidence_gauge

    fig = render_confidence_gauge(0.82)
    assert fig is not None
    mock_plotly_chart.assert_called()


@patch(f"{BASE_PATCH}.metric")
@patch(f"{BASE_PATCH}.columns", return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()))
def test_render_usage_stats(mock_columns, mock_metric) -> None:
    from streamlit_app.components.ui_components import render_usage_stats

    render_usage_stats(10, 1.2, 500, "qwen3.5:2b")
    assert mock_metric.call_count == 4
