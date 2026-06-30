from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import get_service

service = get_service()
settings = service.settings

st.set_page_config(page_title=settings.streamlit.title, layout="wide")

st.title(settings.streamlit.title)
st.caption("Production-grade internet-connected AI agent with LangGraph + Ollama + ChromaDB")

col1, col2, col3 = st.columns(3)

metrics = service.metrics()
monitor = service.monitor()

col1.metric("Completed Runs", int(metrics["counters"].get("agent.completed_runs", 0)))
col2.metric("CPU %", f"{monitor['cpu_percent']:.1f}")
col3.metric("RAM %", f"{monitor['memory']['percent']:.1f}")

st.markdown("## Quick Start")
st.code("""uv sync --all-groups
uv run internet-agent-api
uv run streamlit run streamlit_app/Home.py""")

st.markdown("## Workflow")
st.markdown(
    "User Intent -> Planner -> Search Decision -> Search -> Web Extraction -> "
    "Summarization -> Verification -> Memory -> Reflection -> Report"
)
