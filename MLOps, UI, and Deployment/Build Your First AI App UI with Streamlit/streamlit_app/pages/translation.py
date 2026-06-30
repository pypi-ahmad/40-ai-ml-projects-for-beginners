"""Translation page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ui_components import render_result_box
from streamlit_app.utils.caching import cached_translate
from streamlit_app.utils.helpers import load_sample_text, save_prompt, validate_text_input


LANGUAGES = [
    "Arabic",
    "Chinese",
    "Dutch",
    "French",
    "German",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
    "Turkish",
    "Urdu",
]


def render() -> None:
    st.title("Translation")

    with st.expander("Learning Module: Translation Workflows", expanded=False):
        st.markdown(
            """
**Definition**: Translation converts text from source language to target language.

**Theory**: Sequence-to-sequence generation learns cross-lingual alignment patterns.

**Motivation**: Product teams localize support content and onboarding copy.

**Real-world example**: Global customer success teams translate ticket responses.

**Visual explanation**: Source text + target language constraint -> translated output.

**Code explanation**: This page uses a specialized translation model for predictable outputs.

**Best practices**: Preserve dates, numbers, names, and legal terms exactly.

**Common mistakes**: Using generic chat prompts that add commentary around translation.
            """
        )

    text = st.text_area("Source text", height=140, placeholder="Enter text to translate...")
    target = st.selectbox("Target language", LANGUAGES, index=3)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load sample", use_container_width=True):
            st.session_state["translate_input"] = load_sample_text("translate")
    with col2:
        run = st.button("Translate", type="primary", use_container_width=True)

    if st.session_state.get("translate_input"):
        text = st.session_state["translate_input"]

    if run:
        error = validate_text_input(text, min_length=4)
        if error:
            st.error(error)
            return

        save_prompt(text)
        model_name = st.session_state["selected_models"].get("translation", "translategemma:4b")

        with st.spinner(f"Translating with {model_name}..."):
            translated = cached_translate(text=text, target_lang=target, model=model_name)

        render_result_box(translated, title=f"Translation ({target})")
