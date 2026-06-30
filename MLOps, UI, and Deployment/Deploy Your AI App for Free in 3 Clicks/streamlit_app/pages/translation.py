"""
Translation page.

Translates text between multiple languages.
Teaches: language selection, API parameters, multilingual handling.
"""

import streamlit as st
from streamlit_app.utils.models import translate_text, SUPPORTED_LANGUAGES
from streamlit_app.components.ui_components import (
    render_text_comparison,
    render_example_buttons,
    render_feedback_buttons,
    render_metrics_row,
)
from streamlit_app.utils.helpers import validate_text_input
from streamlit_app.utils.caching import timed_cache, inference_timer


EXAMPLES = {
    "en": [
        "Welcome to our AI application deployment tutorial. "
        "You will learn how to build and deploy apps for free.",
        "The quick brown fox jumps over the lazy dog.",
    ],
    "es": [
        "La inteligencia artificial está transformando el mundo "
        "y creando nuevas oportunidades para todos.",
    ],
    "fr": [
        "L'apprentissage automatique permet aux ordinateurs "
        "d'apprendre sans être explicitement programmés.",
    ],
}

LANGUAGE_NAMES = {k: v["name"] for k, v in SUPPORTED_LANGUAGES.items()}
LANGUAGE_FLAGS = {k: v.get("flag", "") for k, v in SUPPORTED_LANGUAGES.items()}


@timed_cache(max_age_seconds=600)
@inference_timer
def _run_translation_inference(text: str, source_lang: str, target_lang: str) -> dict:
    """Run translation inference through the Streamlit cache layer."""
    return translate_text(text, source_lang=source_lang, target_lang=target_lang, use_cache=False)


def render():
    st.markdown("# 🌍 Translation")
    st.markdown(
        "Translate text between 20+ languages. Powered by Hugging Face "
        "Inference API with Helsinki-NLP OPUS-MT models."
    )

    with st.expander("📖 How Translation Works", expanded=False):
        st.markdown(
            """
        **Machine Translation** converts text from one language to another.

        **Approach:** Helsinki-NLP OPUS-MT models — specialized transformer
        models trained on parallel corpora for each language pair.

        **Real-world uses:**
        - 🌐 Localization: adapt content for global audiences
        - 💬 Customer support: respond in user's language
        - 📄 Document translation: process multilingual documents
        - 🔍 Cross-lingual search: find content across languages

        **Note:** First translation of each pair loads the model (may be slow).
        Subsequent translations are cached for speed.
        """
        )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox(
            "Source language:",
            options=list(LANGUAGE_NAMES.keys()),
            format_func=lambda x: f"{LANGUAGE_FLAGS.get(x, '')} {LANGUAGE_NAMES[x]}",
            index=list(LANGUAGE_NAMES.keys()).index("en"),
            key="translation_source",
        )
    with col2:
        target_lang = st.selectbox(
            "Target language:",
            options=list(LANGUAGE_NAMES.keys()),
            format_func=lambda x: f"{LANGUAGE_FLAGS.get(x, '')} {LANGUAGE_NAMES[x]}",
            index=list(LANGUAGE_NAMES.keys()).index("es"),
            key="translation_target",
        )

    if source_lang == target_lang:
        st.warning("Source and target languages are the same. Translation will be identical.")
    else:
        model_info = f"{LANGUAGE_FLAGS.get(source_lang, '')} {LANGUAGE_NAMES[source_lang]} → {LANGUAGE_FLAGS.get(target_lang, '')} {LANGUAGE_NAMES[target_lang]}"
        st.caption(f"Model: Helsinki-NLP OPUS-MT ({model_info})")

    st.markdown("---")

    example_col, input_col = st.columns([1, 2])
    with example_col:
        st.markdown("**Try an example:**")
        for lang_code, examples in EXAMPLES.items():
            if lang_code == source_lang:
                for i, ex in enumerate(examples):
                    if st.button(
                        ex[:50] + "...",
                        key=f"trans_example_{lang_code}_{i}",
                        use_container_width=True,
                    ):
                        st.session_state["translation_input"] = ex
                        st.rerun()

    with input_col:
        if "input_text" in st.session_state and st.session_state["input_text"]:
            default_text = st.session_state["input_text"]
        else:
            default_text = ""

        text = st.text_area(
            f"Enter text to translate ({LANGUAGE_NAMES[source_lang]}):",
            value=default_text,
            placeholder="Type or paste text here...",
            height=120,
            key="translation_input",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            translate_clicked = st.button(
                "🌍 Translate", type="primary", use_container_width=True
            )
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state["translation_input"] = ""
                st.session_state.pop("translation_result", None)
                st.rerun()

    st.markdown("---")

    if translate_clicked and text:
        error_msg = validate_text_input(text)
        if error_msg:
            st.error(error_msg)
            st.stop()

        with st.spinner(f"Translating from {LANGUAGE_NAMES[source_lang]} to {LANGUAGE_NAMES[target_lang]}..."):
            result = _run_translation_inference(text, source_lang, target_lang)

        if "error" in result:
            st.error(f"Translation failed: {result['error']}")
            st.stop()

        st.session_state["translation_result"] = result
        st.session_state["last_method"] = result.get("method", "unknown")
        st.session_state["last_inference_time"] = result.get("inference_time")

    if st.session_state.get("translation_result") is not None:
        result = st.session_state["translation_result"]

        translated_text = result.get("translated_text", "")
        inference_time = result.get("inference_time", 0.0)
        method = result.get("method", "unknown")

        metrics = {
            "Source Words": len(text.split()),
            "Translated Words": len(translated_text.split()),
            "Language Pair": f"{LANGUAGE_NAMES[source_lang]} → {LANGUAGE_NAMES[target_lang]}",
            "Method": method,
        }
        render_metrics_row(metrics, columns=4)

        render_text_comparison(
            source=text,
            result=translated_text,
            source_label=f"📄 {LANGUAGE_NAMES[source_lang]}",
            result_label=f"📋 {LANGUAGE_NAMES[target_lang]}",
            source_lang=LANGUAGE_NAMES[source_lang],
            result_lang=LANGUAGE_NAMES[target_lang],
            max_length=2000,
        )

        st.caption(f"⚡ Response time: {inference_time:.2f}s")
        render_feedback_buttons("translation")

    elif translate_clicked and not text:
        st.warning("Please enter some text to translate.")


def show():
    """Backward-compatible alias used by older tests/material."""
    render()
