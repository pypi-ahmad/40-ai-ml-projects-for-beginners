from __future__ import annotations

import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Settings")
settings = platform.settings.model_dump(mode="json")
if settings.get("auth", {}).get("api_keys"):
    settings["auth"]["api_keys"] = {key: "***" for key in settings["auth"]["api_keys"]}
st.json(settings)
