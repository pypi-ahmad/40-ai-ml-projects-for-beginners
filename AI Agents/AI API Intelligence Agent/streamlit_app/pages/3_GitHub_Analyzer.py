from __future__ import annotations

import asyncio
import pandas as pd
import plotly.express as px
import streamlit as st

from api_intel_agent.connectors import ConnectorRegistry

st.title('GitHub Analyzer')
q = st.text_input('Search', 'open source llm')
if st.button('Analyze GitHub'):
    result = asyncio.run(ConnectorRegistry().get('github').execute(q))
    df = pd.DataFrame(result.records)
    if not df.empty:
        top = df.nlargest(15, 'stargazers_count')
        fig = px.bar(top, x='name', y='stargazers_count', title='Top repos by stars')
        st.plotly_chart(fig, width='stretch')
        st.dataframe(top[['name', 'stargazers_count', 'language']], width='stretch')
    else:
        st.warning('No data returned')
