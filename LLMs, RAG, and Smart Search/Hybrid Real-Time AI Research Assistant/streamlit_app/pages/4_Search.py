from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from streamlit_app.utils import get_runtime, init_state, mode_from_label

init_state()
runtime = get_runtime()

st.title("Search")
st.caption("Single-turn search over local, web, or hybrid retrieval")

query = st.text_input("Question")
mode_label = st.radio("Mode", ["Auto", "Local", "Web", "Hybrid"], horizontal=True)
prompt_name = st.selectbox(
    "Prompt",
    ["strict_qa", "research_assistant", "teacher", "technical_mentor", "summarizer"],
    index=1,
)
provider = st.selectbox("Web provider", ["duckduckgo", "tavily", "brave"], index=0)

if st.button("Run Search", type="primary") and query.strip():
    with st.spinner("Running workflow..."):
        response = runtime.workflow.ask(
            query=query,
            mode=mode_from_label(mode_label),
            prompt_name=prompt_name,
            provider=provider,
        )
    st.session_state["last_response"] = response
    st.markdown("### Answer")
    st.write(response.answer)
    st.markdown("### Route")
    st.json({"mode": response.mode.value, "reason": response.route_reason})
    st.markdown("### Timings (ms)")
    st.json(asdict(response.timings))
