"""Shared Streamlit helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from hybrid_research_assistant.app import AppRuntime, build_runtime
from hybrid_research_assistant.schemas import RetrievalMode


@st.cache_resource
def get_runtime() -> AppRuntime:
    """Create one runtime instance per Streamlit session."""

    return build_runtime()


def init_state() -> None:
    """Initialize shared Streamlit state."""

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("dark_mode", True)


def response_to_frame(response) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    rows = [
        {
            "source_file": citation.source_file,
            "page_number": citation.page_number,
            "url": citation.url,
            "chunk_id": citation.chunk_id,
            "confidence": citation.confidence,
            "title": citation.title,
        }
        for citation in response.citations
    ]
    return pd.DataFrame(rows)


def save_uploaded_file(uploaded) -> Path:  # type: ignore[no-untyped-def]
    runtime = get_runtime()
    target = runtime.settings.paths.documents_dir / uploaded.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded.getvalue())
    return target


def mode_from_label(label: str) -> RetrievalMode:
    mapping = {
        "Auto": RetrievalMode.AUTO,
        "Local": RetrievalMode.LOCAL,
        "Web": RetrievalMode.WEB,
        "Hybrid": RetrievalMode.HYBRID,
    }
    return mapping[label]
