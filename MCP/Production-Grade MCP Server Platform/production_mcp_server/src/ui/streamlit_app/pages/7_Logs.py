from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Logs")

log_path = Path(platform.settings.logging.file_path)
if log_path.exists():
    content = log_path.read_text(encoding="utf-8")
    st.code("\n".join(content.splitlines()[-300:]))
else:
    st.info("No logs yet")
