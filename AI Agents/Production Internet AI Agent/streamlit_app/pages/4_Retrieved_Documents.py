from __future__ import annotations

import streamlit as st

st.title("Retrieved Documents")
payload = st.session_state.get("last_search")

if not payload:
    st.info("Run Search Explorer first.")
else:
    for idx, doc in enumerate(payload.get("documents", []), start=1):
        with st.expander(f"{idx}. {doc.get('title', doc.get('url', 'document'))}"):
            st.write(doc.get("url", ""))
            st.text_area("Content", value=doc.get("content", "")[:4000], height=220, key=f"doc_{idx}")
