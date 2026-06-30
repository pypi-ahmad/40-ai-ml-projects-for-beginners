"""Streamlit dashboard home page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_intel_agent.memory import MemoryManager

st.set_page_config(page_title="AI API Intelligence Dashboard", layout="wide")

st.title("Production-Grade AI API Intelligence Agent")
st.caption("LangGraph + Ollama + FastAPI + Streamlit")

with st.sidebar:
    st.header("Settings")
    dark_mode = st.toggle("Dark mode", value=True)
    st.write("Dark mode", "enabled" if dark_mode else "disabled")

memory = MemoryManager()
items = memory.history(limit=10)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Recent Analyses")
    if items:
        df = pd.DataFrame([item.model_dump(mode="json") for item in items])
        st.dataframe(df, width='stretch')
    else:
        st.info("No analyses yet. Use API or CLI to generate runs.")

with col2:
    st.subheader("System")
    st.metric("Memory records", len(items))
    st.metric("Pages", 8)
    st.metric("Status", "Ready")
