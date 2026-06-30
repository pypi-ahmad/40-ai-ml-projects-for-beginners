"""Top-level Streamlit application router."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.pages import benchmark, chat, classification, home, ocr_analysis, sentiment, summarization, translation
from streamlit_app.utils.helpers import apply_theme_css, ensure_output_dirs, init_session_state


PAGE_RENDERERS = {
    "Home": home.render,
    "Sentiment Analysis": sentiment.render,
    "Text Summarization": summarization.render,
    "Text Classification": classification.render,
    "Translation": translation.render,
    "Local LLM Chat": chat.render,
    "PDF / OCR Analysis": ocr_analysis.render,
    "Benchmark Dashboard": benchmark.render,
}


def run_app() -> None:
    """Render whole Streamlit AI application with page routing."""
    st.set_page_config(
        page_title="AI Application Studio",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ensure_output_dirs()
    init_session_state()

    selected_page = render_sidebar()
    apply_theme_css(st.session_state.get("theme", "light"))

    renderer = PAGE_RENDERERS.get(selected_page, home.render)
    try:
        renderer()
    except Exception as exc:  # pragma: no cover - UI path
        st.error("An unexpected page-level error occurred.")
        st.exception(exc)


if __name__ == "__main__":
    run_app()
