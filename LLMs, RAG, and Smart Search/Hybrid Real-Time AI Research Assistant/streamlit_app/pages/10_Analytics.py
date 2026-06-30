from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.utils import get_runtime, init_state

init_state()
runtime = get_runtime()

st.title("Analytics")
st.caption("Latency and usage analytics from latest interaction")

response = st.session_state.get("last_response")
if not response:
    st.info("Run a query first to populate analytics.")
else:
    timings = asdict(response.timings)
    frame = pd.DataFrame({"metric": list(timings.keys()), "ms": list(timings.values())})
    st.dataframe(frame, use_container_width=True)
    fig = px.bar(frame, x="metric", y="ms", title="Latency Breakdown")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Current Index Stats")
st.json({"vector_count": runtime.vector_store.count(), "collection": runtime.settings.active_collection_name})
