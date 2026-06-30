from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Monitoring")
metrics = platform.metrics.recent(limit=500)
if not metrics:
    st.info("No metrics available")
else:
    df = pd.DataFrame(metrics)
    st.dataframe(df, use_container_width=True)
    for metric_name in sorted(df["metric_name"].unique()):
        subset = df[df["metric_name"] == metric_name]
        fig = px.line(subset, x="created_at", y="metric_value", title=metric_name)
        st.plotly_chart(fig, use_container_width=True)
