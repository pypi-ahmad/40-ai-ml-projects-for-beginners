from __future__ import annotations

import streamlit as st

from hybrid_research_assistant.schemas import RetrievalMode
from streamlit_app.utils import get_runtime, init_state

init_state()
runtime = get_runtime()

st.title("Chat")
st.caption("Conversational assistant with short-term memory")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a follow-up question")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    runtime.memory.add("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = runtime.workflow.ask(prompt, mode=RetrievalMode.AUTO)
        runtime.memory.add("assistant", response.answer)
        st.markdown(response.answer)
        st.caption(f"mode={response.mode.value} | latency={response.timings.total_ms:.1f} ms")

    st.session_state["messages"].append({"role": "assistant", "content": response.answer})
    st.session_state["last_response"] = response
