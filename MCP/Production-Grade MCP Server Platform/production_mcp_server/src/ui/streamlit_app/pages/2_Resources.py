from __future__ import annotations

import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Resources")
resources = platform.resources.list()
st.dataframe(resources, use_container_width=True)

uri = st.text_input("Read resource URI", value="config://default")
if st.button("Read"):
    try:
        payload = platform.resources.read(uri)
        st.code(payload["content"])
    except Exception as exc:
        st.error(str(exc))
