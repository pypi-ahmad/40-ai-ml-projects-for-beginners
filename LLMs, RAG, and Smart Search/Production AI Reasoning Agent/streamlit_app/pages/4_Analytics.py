"""Analytics dashboards page."""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st

from reasoning_agent.settings import load_settings
from reasoning_agent.utils.json_utils import loads

st.title("Analytics")
settings = load_settings()
report = Path(settings.benchmark_output_dir) / "benchmark_summary.json"

if not report.exists():
    st.warning("Run benchmark first: `reasoning-agent benchmark`.")
else:
    summary = loads(report.read_text(encoding="utf-8"))
    st.dataframe(summary, use_container_width=True)

    fig1 = px.bar(summary, x="model", y="accuracy", title="Hybrid Accuracy")
    fig2 = px.bar(summary, x="model", y="avg_latency_ms", title="Average Latency")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
