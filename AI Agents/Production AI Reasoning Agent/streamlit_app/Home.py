"""Home page for Production AI Reasoning Agent Streamlit app."""

from __future__ import annotations

import streamlit as st

from reasoning_agent.ui.session import ensure_session_state

st.set_page_config(
    page_title="Production AI Reasoning Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_session_state()

st.title("Production AI Reasoning Agent")
st.caption("LangGraph + Ollama + Tool Registry + Chroma Memory + Observability")

st.markdown(
    """
    This interface exposes:
    - Chat reasoning with tool orchestration
    - Execution trace and tool-level telemetry
    - Memory inspection
    - Benchmark analytics and latency views
    - Runtime settings and exports
    """
)

left, right = st.columns(2)
left.metric("Conversations", len(st.session_state.chat_history))
right.metric("Agent Runs", len(st.session_state.run_history))

st.info("Use sidebar pages to start chat, inspect traces, and view analytics.")
