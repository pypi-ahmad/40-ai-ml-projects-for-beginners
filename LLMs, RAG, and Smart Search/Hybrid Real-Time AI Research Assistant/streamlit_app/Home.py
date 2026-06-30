"""Home page for Hybrid Real-Time AI Research Assistant."""

from __future__ import annotations

import streamlit as st

from streamlit_app.utils import get_runtime, init_state

st.set_page_config(
    page_title="Hybrid Real-Time AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)

init_state()
runtime = get_runtime()

st.title("Hybrid Real-Time AI Research Assistant")
st.caption("Local-first LangGraph RAG with ChromaDB + Ollama + optional live web search")

with st.sidebar:
    st.header("Runtime")
    st.write(f"Profile: `{runtime.settings.profile}`")
    st.write(f"LLM: `{runtime.settings.models.primary_llm}`")
    st.write(f"Judge: `{runtime.settings.models.judge_llm}`")
    st.write(f"Embedding: `{runtime.settings.models.embedding_default}`")
    dark_mode = st.toggle("Dark mode", value=st.session_state.get("dark_mode", True))
    st.session_state["dark_mode"] = dark_mode

metrics_col_1, metrics_col_2, metrics_col_3, metrics_col_4 = st.columns(4)
metrics_col_1.metric("Indexed vectors", runtime.vector_store.count())
metrics_col_2.metric("Collection", runtime.settings.active_collection_name)
metrics_col_3.metric("Web provider", runtime.settings.web_search.provider_default)
metrics_col_4.metric("Top-K", runtime.settings.retrieval.top_k_default)

st.markdown("### Quick Start")
st.markdown(
    """
1. Open **Upload Documents** page to add PDFs/DOCX/TXT/MD/HTML files.
2. Open **Index Builder** page and run index build/update.
3. Open **Chat** or **Search** pages to ask questions.
4. Inspect **Retrieved Chunks**, **Sources**, and **Analytics** for transparency.
5. Run **Evaluation** for benchmark metrics.
"""
)
