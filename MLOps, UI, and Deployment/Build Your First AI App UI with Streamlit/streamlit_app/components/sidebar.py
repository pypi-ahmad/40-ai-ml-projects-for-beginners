"""Sidebar navigation and global controls."""

from __future__ import annotations

import streamlit as st

from streamlit_app.config import APP_CONFIG
from streamlit_app.utils.models import list_available_models


PAGES: list[tuple[str, str, str]] = [
    ("Home", "Home", "Architecture, learning path, model status"),
    ("Sentiment Analysis", "Sentiment", "Structured sentiment inference"),
    ("Text Summarization", "Summarize", "Abstractive summaries with word budget"),
    ("Text Classification", "Classify", "Zero-shot category prediction"),
    ("Translation", "Translate", "Local translation mini-app"),
    ("Local LLM Chat", "Chat", "Stateful chat with local models"),
    ("PDF / OCR Analysis", "OCR", "Document extraction + analysis"),
    ("Benchmark Dashboard", "Benchmark", "Latency, memory, quality comparisons"),
]


def _safe_model_options() -> list[str]:
    options = list_available_models()
    if options:
        return options
    return [
        APP_CONFIG.models.chat,
        APP_CONFIG.models.chat_fast,
        APP_CONFIG.models.sentiment,
        APP_CONFIG.models.summarization,
        APP_CONFIG.models.translation,
        APP_CONFIG.models.ocr_primary,
        APP_CONFIG.models.ocr_fallback,
        APP_CONFIG.models.benchmark_extra,
    ]


def render_sidebar() -> str:
    """Render navigation and global settings, then return selected page."""
    with st.sidebar:
        st.title("AI App Studio")
        st.caption("Build production-grade local AI interfaces with Streamlit")

        labels = [label for _, label, _ in PAGES]
        label_to_page = {label: page for page, label, _ in PAGES}
        page_to_label = {page: label for page, label, _ in PAGES}

        current_page = st.session_state.get("page", "Home")
        default_index = labels.index(page_to_label.get(current_page, "Home"))
        selected_label = st.radio("Navigation", labels, index=default_index)
        selected_page = label_to_page[selected_label]
        st.session_state["page"] = selected_page

        st.divider()
        st.subheader("Global Model Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["model_settings"].get("temperature", APP_CONFIG.default_temperature)),
            step=0.05,
            help="Lower values improve consistency for structured tasks.",
        )
        st.session_state["model_settings"]["temperature"] = temperature

        max_tokens = st.slider(
            "Max tokens",
            min_value=128,
            max_value=2048,
            value=int(st.session_state["model_settings"].get("max_tokens", APP_CONFIG.default_max_tokens)),
            step=64,
        )
        st.session_state["model_settings"]["max_tokens"] = max_tokens

        theme = st.selectbox(
            "Theme",
            options=["light", "dark"],
            index=0 if st.session_state.get("theme", "light") == "light" else 1,
        )
        st.session_state["theme"] = theme

        st.divider()
        st.subheader("Model Routing")
        options = _safe_model_options()
        st.session_state["selected_models"]["chat"] = st.selectbox(
            "Chat model",
            options=options,
            index=options.index(st.session_state["selected_models"]["chat"])
            if st.session_state["selected_models"]["chat"] in options
            else 0,
            key="chat_model_selector",
        )

        st.session_state["selected_models"]["chat_fast"] = st.selectbox(
            "Fast chat model",
            options=options,
            index=options.index(APP_CONFIG.models.chat_fast) if APP_CONFIG.models.chat_fast in options else 0,
            key="chat_fast_model_selector",
        )

        st.session_state["selected_models"]["sentiment"] = st.selectbox(
            "Sentiment model",
            options=options,
            index=options.index(st.session_state["selected_models"]["sentiment"])
            if st.session_state["selected_models"]["sentiment"] in options
            else 0,
            key="sentiment_model_selector",
        )

        st.session_state["selected_models"]["summarization"] = st.selectbox(
            "Summarization model",
            options=options,
            index=options.index(st.session_state["selected_models"]["summarization"])
            if st.session_state["selected_models"]["summarization"] in options
            else 0,
            key="summarization_model_selector",
        )

        st.session_state["selected_models"]["classification"] = st.selectbox(
            "Classification model",
            options=options,
            index=options.index(st.session_state["selected_models"]["classification"])
            if st.session_state["selected_models"]["classification"] in options
            else 0,
            key="classification_model_selector",
        )

        st.session_state["selected_models"]["translation"] = st.selectbox(
            "Translation model",
            options=options,
            index=options.index(st.session_state["selected_models"]["translation"])
            if st.session_state["selected_models"]["translation"] in options
            else 0,
            key="translation_model_selector",
        )

        st.session_state["selected_models"]["ocr_primary"] = st.selectbox(
            "OCR primary model",
            options=options,
            index=options.index(st.session_state["selected_models"]["ocr_primary"])
            if st.session_state["selected_models"]["ocr_primary"] in options
            else 0,
            key="ocr_primary_model_selector",
        )

        st.session_state["selected_models"]["ocr_fallback"] = st.selectbox(
            "OCR fallback model",
            options=options,
            index=options.index(st.session_state["selected_models"]["ocr_fallback"])
            if st.session_state["selected_models"]["ocr_fallback"] in options
            else 0,
            key="ocr_fallback_model_selector",
        )

        show_notes = st.checkbox(
            "Show teaching notes",
            value=bool(st.session_state["user_preferences"].get("show_teaching_notes", True)),
        )
        st.session_state["user_preferences"]["show_teaching_notes"] = show_notes

        st.divider()
        st.subheader("Saved Prompts")
        saved_prompts: list[str] = st.session_state.get("saved_prompts", [])
        if not saved_prompts:
            st.caption("No saved prompts yet.")
        else:
            for idx, prompt in enumerate(saved_prompts[:5], start=1):
                st.caption(f"{idx}. {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

        if st.button("Clear chat history", use_container_width=True):
            st.session_state["chat_history"] = []

        st.caption("Local inference only. No cloud API calls.")

    return selected_page
