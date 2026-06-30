from __future__ import annotations

import httpx
import streamlit as st

st.title("Live Workflow")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")
run_id = st.text_input("Run ID")

if st.button("Load Workflow") and run_id:
    try:
        data = httpx.get(f"{api_url}/analytics", params={"run_id": run_id}, timeout=20).json()
        graph_path = data.get("workflow_graph_path")
        st.json(data)
        if graph_path:
            st.info(f"Graph HTML: {graph_path}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load workflow: {exc}")
