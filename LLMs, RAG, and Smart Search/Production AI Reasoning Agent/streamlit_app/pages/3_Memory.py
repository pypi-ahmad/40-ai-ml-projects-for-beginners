"""Memory search page."""

from __future__ import annotations

import streamlit as st

from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.settings import load_settings

st.title("Memory")
settings = load_settings()

if "memory_runner" not in st.session_state:
    st.session_state.memory_runner = AgentRunner(settings=settings)

query = st.text_input("Semantic memory query")
top_k = st.slider("Top K", 1, 20, 5)

if st.button("Search memory") and query:
    hits = st.session_state.memory_runner.artifacts.memory.retrieve(query, k=top_k)
    if not hits:
        st.info("No memory hits.")
    for hit in hits:
        st.markdown(f"Score `{hit.score:.3f}`")
        st.code(hit.text)
        st.json(hit.metadata)
