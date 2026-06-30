"""Chat page."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from reasoning_agent.ui import append_chat, append_run, ensure_session_state, run_agent_query

ensure_session_state()
st.title("Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask question requiring planning, tools, memory, or multi-step reasoning")
if prompt:
    append_chat("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Agent reasoning..."):
        result = run_agent_query(prompt, session_id="streamlit-session")

    answer = result.get("answer", "")
    append_chat("assistant", answer)
    append_run(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": prompt,
            "result": result,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("Reasoning Trace"):
            st.json(
                {
                    "plan": result.get("plan", []),
                    "tool_calls": result.get("tool_calls", []),
                    "observations": result.get("observations", []),
                    "reflection": result.get("reflection", ""),
                    "metrics": result.get("metrics", {}),
                }
            )
