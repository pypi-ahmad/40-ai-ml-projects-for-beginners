"""Runtime settings page."""

from __future__ import annotations

import streamlit as st

from reasoning_agent.settings import load_settings

st.title("Settings")
settings = load_settings()

st.json(settings.model_dump())
st.caption("Edit `.env` or `configs/config.yaml` to change runtime behavior.")
