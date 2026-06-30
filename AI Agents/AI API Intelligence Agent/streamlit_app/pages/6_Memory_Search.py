from __future__ import annotations

import streamlit as st

from api_intel_agent.core.schemas import MemorySearchRequest
from api_intel_agent.memory import MemoryManager

st.title('Memory Search')
query = st.text_input('Semantic query', 'llama')
if st.button('Search memory'):
    hits = MemoryManager().search(MemorySearchRequest(query=query, top_k=10))
    if not hits:
        st.info('No semantic hits.')
    for hit in hits:
        st.write(f"**{hit.id}** score={hit.score:.3f}")
        st.write(hit.content)
        st.json(hit.metadata)
