"""Execution trace page."""

from __future__ import annotations

import streamlit as st

from reasoning_agent.ui import ensure_session_state, selected_run

ensure_session_state()
st.title("Execution Trace")

if not st.session_state.run_history:
    st.warning("No runs yet. Open Chat page and submit prompt.")
    st.stop()

labels = [
    f"{idx + 1}. {run['query'][:60]}"
    for idx, run in enumerate(st.session_state.run_history)
]
index = st.selectbox("Select run", options=range(len(labels)), format_func=lambda i: labels[i])
st.session_state.selected_run_index = index
run = selected_run()
assert run is not None

result = run["result"]
st.subheader("Plan")
st.json(result.get("plan", []))

st.subheader("Observations")
for idx, observation in enumerate(result.get("observations", []), start=1):
    st.markdown(f"{idx}. {observation}")

st.subheader("Reflection")
st.write(result.get("reflection", ""))

st.subheader("Final Answer")
st.success(result.get("answer", ""))
