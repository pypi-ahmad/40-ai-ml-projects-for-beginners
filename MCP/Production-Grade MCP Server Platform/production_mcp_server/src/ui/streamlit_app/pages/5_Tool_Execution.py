from __future__ import annotations

import asyncio
import json

import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Tool Execution")

tool_name = st.selectbox("Tool", options=platform.tools.names())
args = st.text_area("Arguments JSON", value="{}")

if st.button("Run Tool"):
    payload = json.loads(args)
    result = asyncio.run(platform.call_tool(tool_name, payload, session_id="streamlit"))
    st.json(result)
