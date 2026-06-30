"""Streamlit home/chat page."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.schemas import ReasoningMode
from reasoning_agent.settings import load_settings

st.set_page_config(page_title="Production AI Reasoning Agent", page_icon="🤖", layout="wide")

settings = load_settings()

st.title("Production AI Reasoning Agent")
st.caption("LangGraph + Ollama + Dynamic Tool Registry + Memory + Observability")

if "runner" not in st.session_state:
    st.session_state.runner = AgentRunner(settings=settings)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Reasoning mode",
        options=[m.value for m in ReasoningMode],
        index=0,
    )
    session_id = st.text_input("Session ID", value="streamlit-session")
    show_trace = st.toggle("Show reasoning trace", value=True)

prompt = st.chat_input("Ask complex question requiring tools, planning, and reasoning...")
if prompt:
    response = st.session_state.runner.run(session_id=session_id, user_input=prompt, mode=ReasoningMode(mode))
    st.session_state.chat_history.append(
        {
            "user": prompt,
            "assistant": response.answer,
            "trace": [item.model_dump() for item in response.trace],
            "metrics": response.metrics,
            "termination": response.termination_reason,
            "run_id": response.run_id,
        }
    )

for row in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(row["user"])
    with st.chat_message("assistant"):
        st.write(row["assistant"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Termination", row["termination"])
        c2.metric("Reasoning depth", len(row["trace"]))
        c3.metric("Retries", int(row["metrics"].get("retries", 0)))

        if show_trace:
            with st.expander("Execution trace", expanded=False):
                st.json(row["trace"])

st.divider()
st.subheader("Downloads")
if st.session_state.chat_history:
    export = json.dumps(st.session_state.chat_history, indent=2)
    st.download_button(
        "Download conversation JSON",
        data=export,
        file_name="conversation_export.json",
        mime="application/json",
    )

trace_dir = Path(settings.log_dir) / "traces"
st.caption(f"Trace directory: `{trace_dir}`")
