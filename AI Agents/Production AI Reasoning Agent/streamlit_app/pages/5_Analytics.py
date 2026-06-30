"""Analytics dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reasoning_agent.ui import ensure_session_state

ensure_session_state()
st.title("Analytics")

if not st.session_state.run_history:
    st.warning("No run data yet.")
    st.stop()

rows = []
for run in st.session_state.run_history:
    metrics = run["result"].get("metrics", {})
    rows.append(
        {
            "query": run["query"],
            "latency_ms": metrics.get("total_latency_ms", 0.0),
            "iterations": metrics.get("iterations", 0.0),
            "tool_calls": metrics.get("tool_calls", 0.0),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Avg Latency (ms)", round(float(df["latency_ms"].mean()), 2))
c2.metric("Avg Iterations", round(float(df["iterations"].mean()), 2))
c3.metric("Avg Tool Calls", round(float(df["tool_calls"].mean()), 2))

st.subheader("Latency Trend")
st.line_chart(df["latency_ms"])
