from __future__ import annotations

import httpx
import streamlit as st

st.title("Agents")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")

try:
    agents = httpx.get(f"{api_url}/agents", timeout=20).json().get("agents", [])
    st.dataframe(agents, use_container_width=True)
except Exception as exc:  # noqa: BLE001
    st.error(f"Unable to fetch agents: {exc}")
