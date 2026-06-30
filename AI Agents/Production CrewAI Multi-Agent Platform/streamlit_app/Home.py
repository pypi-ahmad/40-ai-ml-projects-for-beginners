"""Streamlit dashboard home page."""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="Crew Platform Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.title("Production CrewAI Multi-Agent Platform")
st.caption("Enterprise multi-agent collaboration dashboard")

api_default = os.getenv("CREW_PLATFORM_API_URL", "http://127.0.0.1:8000")
if "api_url" not in st.session_state:
    st.session_state["api_url"] = api_default

st.session_state["api_url"] = st.text_input("FastAPI URL", st.session_state["api_url"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mode", "Plan Approval Required")
with col2:
    st.metric("Parallel Tasks", "2")
with col3:
    st.metric("Consensus", "Conditional")

st.markdown(
    """
### Navigation
Use left sidebar pages:
- Dashboard
- Crew Monitor
- Live Workflow
- Agents
- Tasks
- Memory
- Knowledge Base
- Reports
- Analytics
- Settings
"""
)
