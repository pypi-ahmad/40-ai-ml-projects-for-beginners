"""Settings and export page."""

from __future__ import annotations

import json

import streamlit as st

from reasoning_agent.config import get_settings
from reasoning_agent.ui import ensure_session_state

ensure_session_state()
st.title("Settings")

st.session_state.ui_dark_mode = st.toggle("Dark mode", value=st.session_state.ui_dark_mode)

st.subheader("Export")
chat_json = json.dumps(st.session_state.chat_history, indent=2)
runs_json = json.dumps(st.session_state.run_history, indent=2)

st.download_button(
    "Download Conversation JSON",
    data=chat_json,
    file_name="conversation.json",
    mime="application/json",
)

st.download_button(
    "Download Runs JSON",
    data=runs_json,
    file_name="run_history.json",
    mime="application/json",
)

st.caption("Model and tool runtime settings are controlled through YAML/.env config.")

settings = get_settings()
st.subheader("Runtime Snapshot")
st.json(
    {
        "runtime_mode": settings.agent.runtime_mode,
        "reasoning_mode": settings.agent.reasoning_mode,
        "max_iterations": settings.agent.max_iterations,
        "graph_timeout_seconds": settings.agent.graph_timeout_seconds,
        "chroma_enabled": settings.memory.chroma_enabled,
        "memory_top_k": settings.memory.memory_top_k,
        "python_tool_enabled": settings.tools.enable_python_tool,
        "enabled_tools": settings.tools.enabled_tools,
        "optional_tools": settings.tools.optional_tools,
        "primary_model": settings.llm.primary_model,
    }
)
