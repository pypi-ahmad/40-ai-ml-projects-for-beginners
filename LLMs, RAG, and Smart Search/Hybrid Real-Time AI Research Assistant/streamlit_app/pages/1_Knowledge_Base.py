from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.utils import get_runtime, init_state

init_state()
runtime = get_runtime()

st.title("Knowledge Base")
st.caption("Inspect local knowledge base and indexed document metadata")

st.metric("Indexed vectors", runtime.vector_store.count())
rows = runtime.vector_store.list_documents()
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No indexed documents yet.")
