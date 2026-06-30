from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Reports")

query = st.text_input("Workflow query", value="Search docs then create report")
if st.button("Generate"):
    result = asyncio.run(platform.run_workflow(query))
    st.json(result)

reports_dir = Path(platform.settings.plugins.directory).parents[0] / "reports"
reports = sorted(str(p) for p in reports_dir.glob("*"))
st.write(reports)
