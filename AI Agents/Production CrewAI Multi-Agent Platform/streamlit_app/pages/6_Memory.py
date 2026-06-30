from __future__ import annotations

import httpx
import streamlit as st

st.title("Memory")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")
query = st.text_input("Semantic query")

if st.button("Load Memory"):
    try:
        data = httpx.get(f"{api_url}/memory", params={"limit": 50, "query": query or None}, timeout=20).json()
        st.json(data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to fetch memory: {exc}")
