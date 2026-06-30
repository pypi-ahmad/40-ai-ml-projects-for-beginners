"""Execution trace explorer page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from reasoning_agent.settings import load_settings
from reasoning_agent.utils.json_utils import loads

st.title("Execution Trace")
settings = load_settings()
trace_dir = Path(settings.log_dir) / "traces"
trace_dir.mkdir(parents=True, exist_ok=True)

files = sorted(trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not files:
    st.info("No traces yet. Run chat first.")
else:
    selected = st.selectbox("Run trace", options=files, format_func=lambda p: p.name)
    payload = loads(selected.read_text(encoding="utf-8"))
    st.json(payload)
