from __future__ import annotations

import httpx
import streamlit as st

st.title("Crew Monitor")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")

query = st.text_area("Objective", "Analyze AI agent platform readiness for enterprise rollout")
if st.button("Plan + Execute"):
    try:
        planned = httpx.post(f"{api_url}/crew", json={"query": query, "auto_execute": False}, timeout=30).json()
        run_id = planned["run_id"]
        httpx.post(f"{api_url}/crew/{run_id}/approve", json={"approved": True, "reviewer": "dashboard"}, timeout=20)
        result = httpx.post(f"{api_url}/crew/{run_id}/execute", timeout=120).json()
        st.success(f"Run completed: {run_id}")
        st.json(result)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Execution failed: {exc}")
