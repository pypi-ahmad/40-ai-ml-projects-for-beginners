from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import get_service

service = get_service()

st.title("Dashboard")
metrics = service.metrics()
monitor = service.monitor()

st.subheader("Runtime")
col1, col2, col3 = st.columns(3)
col1.metric("Tool Calls", int(sum(v for k, v in metrics["counters"].items() if k.startswith("tool."))))
col2.metric("Cache Hits", int(metrics["counters"].get("cache.search.hit", 0)))
col3.metric("Cache Miss", int(metrics["counters"].get("cache.search.miss", 0)))

st.subheader("System")
st.json(monitor)
