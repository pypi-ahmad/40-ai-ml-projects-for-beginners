from __future__ import annotations

import asyncio
import pandas as pd
import streamlit as st

from api_intel_agent.connectors import ConnectorRegistry

st.title('News Analyzer')
q = st.text_input('Topic', 'AI')
if st.button('Analyze News'):
    result = asyncio.run(ConnectorRegistry().get('news').execute(q))
    df = pd.DataFrame(result.records)
    if not df.empty:
        st.dataframe(df[['title', 'source', 'publishedAt']] if 'publishedAt' in df.columns else df.head(20), width='stretch')
    else:
        st.warning('No news data returned')
