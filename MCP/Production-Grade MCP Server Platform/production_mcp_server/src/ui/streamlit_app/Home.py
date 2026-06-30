from __future__ import annotations

import streamlit as st

from ui.service import get_platform

st.set_page_config(page_title="Production MCP Dashboard", layout="wide")
platform = get_platform()

st.title("Production MCP Server Platform")
st.caption("Operations dashboard for tools, resources, prompts, memory, workflows, metrics, and logs")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tools", len(platform.tools.names()))
c2.metric("Resources", len(platform.resources.list()))
c3.metric("Prompts", len(platform.prompts.names()))
c4.metric("Runtime", platform.settings.transport.runtime)

st.subheader("Recent Metrics")
metrics = platform.metrics.recent(limit=50)
st.dataframe(metrics, use_container_width=True)
