from __future__ import annotations

import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Prompt Library")

prompts = platform.prompts.list()
st.dataframe(prompts, use_container_width=True)

prompt_name = st.selectbox("Prompt", options=platform.prompts.names())
variables = st.text_area("Variables JSON", value='{"topic": "MCP"}')
if st.button("Render Prompt"):
    import json

    data = json.loads(variables)
    rendered = platform.prompts.render(prompt_name, data)
    st.code(rendered)
