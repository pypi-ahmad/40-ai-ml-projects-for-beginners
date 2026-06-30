"""Local LLM chat page with session-state history."""

from __future__ import annotations

import streamlit as st

from streamlit_app.services.prompts import chat_system_prompt, prompt_quality_examples
from streamlit_app.utils.caching import cached_chat
from streamlit_app.utils.helpers import add_to_chat_history, save_prompt, trim_chat_history


def _chat_payload() -> list[dict[str, str]]:
    """Build model payload using system prompt + conversation history."""
    history: list[dict[str, str]] = st.session_state.get("chat_history", [])
    return [{"role": "system", "content": chat_system_prompt()}] + history


def render() -> None:
    st.title("Local LLM Chat")

    with st.expander("Learning Module: Stateful Chat in Streamlit", expanded=False):
        st.markdown(
            """
**Definition**: Stateful chat stores conversation turns across reruns.

**Theory**: Streamlit reruns script top-to-bottom on every interaction, so state must live in `st.session_state`.

**Motivation**: Without state, each user prompt loses prior context.

**Real-world example**: Internal copilots use session history for follow-up questions.

**Visual explanation**: user message -> append session state -> model call -> append assistant message.

**Code explanation**: This page serializes chat messages into tuples for cache safety.

**Best practices**: Keep bounded history window to control latency and token usage.

**Common mistakes**: Appending huge raw documents into chat context without truncation.
            """
        )

    model_name = st.session_state["selected_models"].get("chat", "qwen3.5:4b")
    temperature = float(st.session_state["model_settings"].get("temperature", 0.2))
    max_tokens = int(st.session_state["model_settings"].get("max_tokens", 512))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model", model_name)
    with c2:
        st.metric("Temperature", f"{temperature:.2f}")
    with c3:
        st.metric("Max tokens", str(max_tokens))

    if st.session_state["user_preferences"].get("show_prompt_examples", True):
        examples = prompt_quality_examples()
        with st.expander("Prompt design: good vs bad", expanded=False):
            st.code(f"Bad: {examples['bad']}\n\nGood: {examples['good']}")

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about AI app design, ML deployment, or Streamlit patterns")
    if prompt:
        save_prompt(prompt)
        add_to_chat_history("user", prompt)
        trim_chat_history(20)

        with st.chat_message("user"):
            st.markdown(prompt)

        payload = _chat_payload()
        tuple_payload = tuple((item["role"], item["content"]) for item in payload)
        with st.chat_message("assistant"):
            with st.spinner("Running local inference..."):
                response = cached_chat(tuple_payload, model_name, temperature, max_tokens)
            st.markdown(response)

        add_to_chat_history("assistant", response)
        trim_chat_history(20)

    clear_col, save_col = st.columns(2)
    with clear_col:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
    with save_col:
        if st.button("Save last prompt", use_container_width=True):
            history = st.session_state.get("chat_history", [])
            if history:
                last_user = [item for item in history if item["role"] == "user"]
                if last_user:
                    save_prompt(last_user[-1]["content"])
                    st.success("Prompt saved in sidebar list.")
