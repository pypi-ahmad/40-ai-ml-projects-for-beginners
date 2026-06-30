from __future__ import annotations

import httpx
import streamlit as st

st.title("Dashboard")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")

try:
    health = httpx.get(f"{api_url}/health", timeout=10).json()
    metrics = httpx.get(f"{api_url}/metrics", timeout=10).json()
    st.json({"health": health, "metrics": metrics})
except Exception as exc:  # noqa: BLE001
    st.error(f"API unavailable: {exc}")
