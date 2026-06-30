from __future__ import annotations

import streamlit as st

from streamlit_app.utils import get_runtime, init_state

init_state()
runtime = get_runtime()

st.title("Settings")
st.caption("Runtime configuration snapshot")

st.json(
    {
        "profile": runtime.settings.profile,
        "models": runtime.settings.models.model_dump(),
        "chunking": runtime.settings.chunking.model_dump(),
        "retrieval": runtime.settings.retrieval.model_dump(),
        "web_search": runtime.settings.web_search.model_dump(),
        "cache": runtime.settings.cache.model_dump(),
    }
)
