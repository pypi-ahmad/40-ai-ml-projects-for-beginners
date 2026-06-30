from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import get_service, run_async

service = get_service()

st.title("Search Explorer")
query = st.text_input("Query", "latest python version")
providers = st.multiselect("Providers", options=["duckduckgo", "news", "wikipedia", "github"], default=["duckduckgo", "news"])

if st.button("Search", type="primary"):
    sid = st.session_state.get("session_id", str(uuid.uuid4())[:8])
    st.session_state["session_id"] = sid
    with st.spinner("Searching and retrieving..."):
        payload = run_async(service.search(session_id=sid, query=query, providers=providers))
        st.session_state["last_search"] = payload

if "last_search" in st.session_state:
    payload = st.session_state["last_search"]
    st.metric("Latency (ms)", f"{payload['latency_ms']:.1f}")
    st.metric("From Cache", str(payload["from_cache"]))
    st.dataframe(payload.get("results", []), use_container_width=True)
