from __future__ import annotations

import httpx
import streamlit as st

st.title("Analytics")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")
run_id = st.text_input("Run ID (optional)")

if st.button("Load Analytics"):
    try:
        params = {"run_id": run_id} if run_id else {}
        data = httpx.get(f"{api_url}/analytics", params=params, timeout=20).json()
        st.json(data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load analytics: {exc}")
