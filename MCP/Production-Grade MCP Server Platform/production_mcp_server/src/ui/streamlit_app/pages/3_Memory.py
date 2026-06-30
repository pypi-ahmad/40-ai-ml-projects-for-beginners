from __future__ import annotations

import streamlit as st

from ui.service import get_platform

platform = get_platform()
st.title("Memory")

query = st.text_input("Semantic query", value="server")
top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
if st.button("Search memory"):
    st.json(platform.memory.semantic_search(query, top_k=top_k))

st.subheader("Recent Conversations")
st.dataframe(platform.memory.fetch_recent_conversations("default", limit=20), use_container_width=True)
