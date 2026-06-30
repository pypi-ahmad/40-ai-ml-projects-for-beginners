from __future__ import annotations

import httpx
import streamlit as st

st.title("Reports")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")
run_id = st.text_input("Run ID")
fmt = st.selectbox("Export format", ["markdown", "json", "html", "pdf"])

if st.button("Fetch Report") and run_id:
    try:
        data = httpx.get(f"{api_url}/reports", params={"run_id": run_id}, timeout=20).json()
        st.json(data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to fetch report: {exc}")

if st.button("Export Report") and run_id:
    try:
        data = httpx.post(f"{api_url}/reports/{run_id}/export", json={"format": fmt}, timeout=30).json()
        st.json(data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Export failed: {exc}")
