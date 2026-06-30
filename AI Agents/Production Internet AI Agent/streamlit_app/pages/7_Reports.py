from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import get_service

service = get_service()

st.title("Reports")
fmt = st.selectbox("Format", ["json", "markdown", "html", "pdf"])
session_id = st.text_input("Session ID", st.session_state.get("session_id", "default"))

if st.button("Generate Report"):
    payload = st.session_state.get("last_chat")
    if not payload:
        st.warning("Run chat first to generate report.")
    else:
        report_payload = payload.get("report", {})
        out = service.export_report(session_id=session_id, payload=report_payload, fmt=fmt)
        st.success(f"Saved: {out['path']}")

reports = service.history(session_id).get("reports", [])
st.dataframe(reports, use_container_width=True)
