"""
Sidebar component for the AI Application.

Provides model selection, navigation, settings, and cache controls.
This teaches the pattern of centralized UI state management.
"""

import streamlit as st
from streamlit_app.utils.config import is_hf_configured


NAV_ITEMS = [
    ("🏠", "Home", "home"),
    ("💬", "Sentiment Analysis", "sentiment"),
    ("📝", "Text Summarization", "summarization"),
    ("🏷️", "Text Classification", "classification"),
    ("🌍", "Translation", "translation"),
]


def render_sidebar():
    """
    Render the sidebar with navigation, model selection, and settings.

    Teaches:
    - Consistent navigation across pages
    - Session state for persistence
    - Cache management in deployed apps
    - Environment variable awareness
    """
    _render_navigation()
    _render_model_info()
    _render_cache_controls()
    _render_environment_status()


def _render_navigation():
    """Render navigation menu."""
    st.sidebar.markdown("## 🧭 Navigation")

    for emoji, label, page_id in NAV_ITEMS:
        if st.sidebar.button(
            f"{emoji} {label}",
            key=f"nav_{page_id}",
            use_container_width=True,
            type="secondary" if st.session_state.get("current_page") != page_id else "primary",
        ):
            st.session_state["current_page"] = page_id
            st.rerun()

    st.sidebar.markdown("---")


def _render_model_info():
    """Show current inference status summary."""
    method = st.session_state.get("last_method", "")
    inference_time = st.session_state.get("last_inference_time", None)

    st.sidebar.markdown("## ⚡ Inference Status")
    if method:
        st.sidebar.info(f"**Last method:** {method}")
    if inference_time is not None:
        st.sidebar.info(f"**Last inference:** {inference_time}s")
    if not method and inference_time is None:
        st.sidebar.caption("No inference yet. Try a task above!")

    st.sidebar.markdown("---")


def _render_cache_controls():
    """Cache management controls for deployment operations."""
    st.sidebar.markdown("## 🗑️ Cache")
    st.sidebar.caption(
        "Caching speeds up repeat requests. "
        "Clear cache to force fresh inference."
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
    with col2:
        if st.button("Reset All", use_container_width=True):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

    st.sidebar.markdown("---")


def _render_environment_status():
    """Show configuration and environment status."""
    st.sidebar.markdown("## 🔧 Environment")

    api_status = "✅ Configured" if is_hf_configured() else "⚠️ Not set (fallback mode)"
    st.sidebar.write(f"**HF API Token:** {api_status}")

    with st.sidebar.expander("About", expanded=False):
        st.markdown(
            """
        **Deploy Your AI App for Free in 3 Clicks**

        Teaches end-to-end AI app deployment:
        1. Build multi-page Streamlit app
        2. Use HF Inference API (no GPU needed)
        3. Deploy to Streamlit Cloud (free)
        4. Monitor and iterate

        See project docs in `README.md` and `docs/`.
        """
        )
