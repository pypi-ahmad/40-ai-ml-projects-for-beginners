"""Tool usage analytics page."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import plotly.express as px
import streamlit as st

from reasoning_agent.settings import load_settings
from reasoning_agent.utils.json_utils import loads

st.title("Tool Calls")
settings = load_settings()
trace_dir = Path(settings.log_dir) / "traces"
trace_dir.mkdir(parents=True, exist_ok=True)

counter: Counter[str] = Counter()
for file in trace_dir.glob("*.json"):
    payload = loads(file.read_text(encoding="utf-8"))
    for row in payload.get("trace", []):
        action = row.get("action") or {}
        tool = action.get("name")
        if tool:
            counter[str(tool)] += 1

if not counter:
    st.info("No tool calls yet.")
else:
    data = [{"tool": k, "count": v} for k, v in counter.items()]
    fig = px.bar(data, x="tool", y="count", title="Tool Usage Frequency")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(data, use_container_width=True)
