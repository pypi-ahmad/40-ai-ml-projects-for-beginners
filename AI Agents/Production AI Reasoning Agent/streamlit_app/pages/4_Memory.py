"""Memory explorer page."""

from __future__ import annotations

import streamlit as st

from reasoning_agent.ui import ensure_session_state

ensure_session_state()
st.title("Memory")

st.subheader("Conversation Memory")
if not st.session_state.chat_history:
    st.info("Conversation memory empty.")
else:
    st.json(st.session_state.chat_history)

st.subheader("Run Memory")
if not st.session_state.run_history:
    st.info("Run memory empty.")
else:
    st.json(st.session_state.run_history[-5:])
