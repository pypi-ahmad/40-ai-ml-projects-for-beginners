from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from hybrid_research_assistant.schemas import ChunkingStrategy
from streamlit_app.utils import get_runtime, init_state

init_state()
runtime = get_runtime()
settings = runtime.settings

st.title("Index Builder")
st.caption("Build/update persistent ChromaDB index")

chunk_size = st.selectbox("Chunk size", settings.chunking.chunk_sizes, index=2)
chunk_overlap = st.selectbox("Chunk overlap", settings.chunking.chunk_overlaps, index=2)
strategy = st.selectbox("Chunking strategy", [item.value for item in ChunkingStrategy], index=0)

col_1, col_2 = st.columns(2)
if col_1.button("Build / Update Index", type="primary"):
    with st.spinner("Indexing documents..."):
        report = runtime.indexer.build_or_update(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=ChunkingStrategy(strategy),
            force_rebuild=False,
        )
    st.success("Indexing complete")
    st.json(asdict(report))

if col_2.button("Rebuild Index"):
    with st.spinner("Rebuilding index..."):
        report = runtime.indexer.build_or_update(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=ChunkingStrategy(strategy),
            force_rebuild=True,
        )
    st.success("Rebuild complete")
    st.json(asdict(report))
