from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

st.title("Settings")
settings_path = Path("configs/settings.yaml")

if settings_path.exists():
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    st.json(raw)
else:
    st.warning("settings.yaml not found")

st.caption("Edit configs/settings.yaml and restart API for changes to apply.")
