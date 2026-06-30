from __future__ import annotations

import streamlit as st

from api_intel_agent.config import load_settings

st.title('Settings')
settings = load_settings()
st.subheader('LLM')
st.json(settings.llm.model_dump(mode='json'))
st.subheader('Cache')
st.json(settings.cache.model_dump(mode='json'))
st.subheader('Memory')
st.json(settings.memory.model_dump(mode='json'))
