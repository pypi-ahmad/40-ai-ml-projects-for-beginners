from __future__ import annotations

import asyncio
import streamlit as st

from api_intel_agent.connectors import ConnectorRegistry

st.title('Live API Explorer')
registry = ConnectorRegistry()
provider = st.selectbox('Provider', registry.list_names())
query = st.text_input('Query', 'ai')

if st.button('Run'):
    result = asyncio.run(registry.get(provider).execute(query))
    st.write(result.status)
    st.json(result.model_dump(mode='json'))
