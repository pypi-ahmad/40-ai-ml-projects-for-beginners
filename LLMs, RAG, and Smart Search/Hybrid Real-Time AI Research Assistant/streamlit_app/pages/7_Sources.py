from __future__ import annotations

import streamlit as st

from streamlit_app.utils import init_state, response_to_frame

init_state()

st.title("Sources")
st.caption("Citations and provenance from last response")

response = st.session_state.get("last_response")
if not response:
    st.info("Run a query in Search or Chat page first.")
else:
    frame = response_to_frame(response)
    if frame.empty:
        st.warning("No citations produced.")
    else:
        st.dataframe(frame, use_container_width=True)
