from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import get_service

service = get_service()

st.title("Memory")
query = st.text_input("Semantic query", "python release")
if st.button("Search Memory"):
    hits = service.memory_search(query=query, top_k=8)
    st.dataframe(hits.get("hits", []), use_container_width=True)

session_id = st.text_input("Session ID", st.session_state.get("session_id", "default"))
if st.button("Load History"):
    history = service.history(session_id)
    st.json(history)
