from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import get_service

service = get_service()

st.title("Analytics")
analytics = service.analytics(st.session_state.get("session_id"))
metrics = analytics["metrics"]

latencies = metrics.get("latencies", {})
if latencies:
    df = [
        {"metric": k, "avg_ms": v["avg_ms"], "max_ms": v["max_ms"], "count": v["count"]}
        for k, v in latencies.items()
    ]
    fig = px.bar(df, x="metric", y="avg_ms", title="Average Latency by Metric")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Most Used Tools")
st.dataframe(
    [{"tool": tool, "count": count} for tool, count in analytics.get("most_used_tools", [])],
    use_container_width=True,
)
