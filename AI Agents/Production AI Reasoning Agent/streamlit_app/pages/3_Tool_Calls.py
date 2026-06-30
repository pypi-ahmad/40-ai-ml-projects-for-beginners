"""Tool calls page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reasoning_agent.ui import ensure_session_state

ensure_session_state()
st.title("Tool Calls")

if not st.session_state.run_history:
    st.info("No tool calls captured yet.")
    st.stop()

all_calls = []
for run in st.session_state.run_history:
    for call in run["result"].get("tool_calls", []):
        all_calls.append(
            {
                "query": run["query"],
                "tool": call.get("tool_name"),
                "success": call.get("success"),
                "error": call.get("error"),
            }
        )

if not all_calls:
    st.info("No tool calls in current runs.")
    st.stop()

df = pd.DataFrame(all_calls)
st.dataframe(df, use_container_width=True)

st.subheader("Tool Frequency")
st.bar_chart(df["tool"].value_counts())
