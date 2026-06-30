from __future__ import annotations

import httpx
import streamlit as st

st.title("Knowledge Base")
api_url = st.session_state.get("api_url", "http://127.0.0.1:8000")

source_path = st.text_input("Local file path to ingest")
source_url = st.text_input("URL to ingest")
query = st.text_input("Search query")

if st.button("Ingest"):
    try:
        payload = {"path": source_path or None, "url": source_url or None}
        data = httpx.post(f"{api_url}/knowledge", json=payload, timeout=60).json()
        st.json(data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ingestion failed: {exc}")

if st.button("Search") and query:
    try:
        data = httpx.get(f"{api_url}/knowledge", params={"q": query, "top_k": 5}, timeout=20).json()
        st.json(data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Knowledge search failed: {exc}")
