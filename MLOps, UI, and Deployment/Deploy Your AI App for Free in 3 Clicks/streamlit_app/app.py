"""
Main Streamlit application entry point.

Auto-discovers pages in pages/ directory and renders them.
Teaches: multi-page app architecture, session state, theme config.
"""

import streamlit as st
from streamlit_app.components.sidebar import render_sidebar


PAGE_REGISTRY = {
    "home": ("🏠 Home", "streamlit_app.pages.home"),
    "sentiment": ("💬 Sentiment Analysis", "streamlit_app.pages.sentiment"),
    "summarization": ("📝 Summarization", "streamlit_app.pages.summarization"),
    "classification": ("🏷️ Classification", "streamlit_app.pages.classification"),
    "translation": ("🌍 Translation", "streamlit_app.pages.translation"),
}


def init_session_state():
    """Initialize persistent session state variables."""
    defaults = {
        "current_page": "home",
        "input_text": "",
        "sentiment_result": None,
        "summarization_result": None,
        "classification_result": None,
        "translation_result": None,
        "last_method": "",
        "last_inference_time": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_page():
    """Configure page settings and layout."""
    st.set_page_config(
        page_title="AI App Deployment Demo",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        .stAppHeader { background-color: transparent; }
        .main > div { padding: 1rem 2rem; }
        .stButton button { border-radius: 8px; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""
    setup_page()
    init_session_state()

    with st.sidebar:
        st.markdown("# 🚀 AI App")
        st.markdown("Deployment Demo\n")
        render_sidebar()

    current_page = st.session_state.get("current_page", "home")

    if current_page in PAGE_REGISTRY:
        _, module_path = PAGE_REGISTRY[current_page]
        module = __import__(module_path, fromlist=["render"])
        if hasattr(module, "render"):
            module.render()
        else:
            st.error(f"Page '{current_page}' has no render() function.")
    else:
        st.error(f"Unknown page: '{current_page}'. Navigate from sidebar.")


if __name__ == "__main__":
    main()
