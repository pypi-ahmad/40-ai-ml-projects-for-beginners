from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import get_service

service = get_service()

st.title("Monitoring")
st.json(service.monitor())
