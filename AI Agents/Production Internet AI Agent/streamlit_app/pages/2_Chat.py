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

st.title("Chat")
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

st.write(f"Session: `{st.session_state['session_id']}`")
query = st.text_area("Ask question", height=120)

if st.button("Run", type="primary") and query.strip():
    with st.spinner("Thinking..."):
        payload = run_async(service.chat(st.session_state["session_id"], query))
        st.session_state["last_chat"] = payload

if "last_chat" in st.session_state:
    payload = st.session_state["last_chat"]
    st.subheader("Answer")
    st.write(payload["answer"])

    st.subheader("Confidence")
    st.metric("Confidence", f"{payload['confidence']:.3f}")

    st.subheader("Sources")
    st.dataframe(payload.get("citations", []), use_container_width=True)

    st.subheader("Reasoning Trace (Tool/Action)")
    st.json(payload.get("reasoning_trace", []))
