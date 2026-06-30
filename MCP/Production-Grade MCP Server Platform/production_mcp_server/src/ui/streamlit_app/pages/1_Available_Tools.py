from __future__ import annotations

import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Available Tools")
st.dataframe(platform.tools.list(), use_container_width=True)
