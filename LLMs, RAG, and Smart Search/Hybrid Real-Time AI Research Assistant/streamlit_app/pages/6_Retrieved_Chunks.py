from __future__ import annotations

import streamlit as st

from streamlit_app.utils import init_state

init_state()

st.title("Retrieved Chunks")
st.caption("Inspect retrieved evidence from last query")

response = st.session_state.get("last_response")
if not response:
    st.info("Run a query in Search or Chat page first.")
else:
    for idx, row in enumerate(response.retrieved, start=1):
        with st.expander(f"{idx}. {row.metadata.get('source', 'unknown')} | score={row.score:.4f}"):
            st.json(row.metadata)
            st.write(row.text)
