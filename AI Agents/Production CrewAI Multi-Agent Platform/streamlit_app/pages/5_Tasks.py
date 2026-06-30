from __future__ import annotations

import httpx
import streamlit as st

st.title("Tasks")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")
run_id = st.text_input("Run ID")

if st.button("Load Tasks") and run_id:
    try:
        data = httpx.get(f"{api_url}/tasks", params={"run_id": run_id}, timeout=20).json()
        st.dataframe(data.get("tasks", []), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to fetch tasks: {exc}")
