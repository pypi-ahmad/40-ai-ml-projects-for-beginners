"""Streamlit dashboard for workflow inspection and control."""

from __future__ import annotations

import json

import streamlit as st

from langgraph_platform.analytics.service import AnalyticsService
from langgraph_platform.config.loader import load_config
from langgraph_platform.engine.workflow import LangGraphWorkflowEngine
from langgraph_platform.ui.graph_visualizer import to_mermaid


def launch_dashboard() -> None:
    """Launch multipage Streamlit dashboard."""

    st.set_page_config(page_title="LangGraph Platform", layout="wide")
    st.title("Production LangGraph Multi-Agent Platform")

    config = load_config()
    engine = LangGraphWorkflowEngine(config)
    analytics = AnalyticsService(engine.sqlite_store)

    page = st.sidebar.selectbox(
        "Page",
        [
            "Dashboard",
            "Workflow Graph",
            "Live Execution",
            "Agents",
            "Shared State",
            "Memory",
            "Knowledge Base",
            "Reports",
            "Analytics",
            "Configuration",
        ],
    )

    if page == "Dashboard":
        st.subheader("System Snapshot")
        st.json(analytics.summary())

    elif page == "Workflow Graph":
        st.subheader("Graph Visualization")
        graph_info = engine.inspect_graph()
        st.code(to_mermaid(graph_info), language="mermaid")
        st.json(graph_info)

    elif page == "Live Execution":
        st.subheader("Run Workflow")
        request = st.text_area("User Request", height=160)
        if st.button("Execute") and request.strip():
            result = engine.run(request)
            st.success(f"Workflow {result.workflow_id} complete")
            st.metric("Confidence", f"{result.confidence:.2f}")
            st.markdown(result.final_report)

    elif page == "Agents":
        from langgraph_platform.agents.registry import list_agents

        st.subheader("Agent Registry")
        for agent in list_agents():
            st.markdown(f"### {agent.role}")
            st.write(
                {
                    "name": agent.name,
                    "objective": agent.objective,
                    "tools": agent.tools,
                    "constraints": agent.constraints,
                    "retry_strategy": agent.retry_strategy,
                }
            )

    elif page == "Shared State":
        st.subheader("Recent Workflow States")
        st.json(engine.sqlite_store.list_recent_runs(limit=10))

    elif page == "Memory":
        st.subheader("Memory Explorer")
        query = st.text_input("Search memory")
        if query:
            result = engine.runtime.tool_registry.run(
                "memory_search", {"query": query, "limit": 10}
            )
            st.json(result.output)

    elif page == "Knowledge Base":
        st.subheader("Ingest Documents")
        paths = st.text_area("Paths (one per line)")
        urls = st.text_area("URLs (one per line)")
        if st.button("Ingest"):
            path_list = [item.strip() for item in paths.splitlines() if item.strip()]
            url_list = [item.strip() for item in urls.splitlines() if item.strip()]
            if path_list:
                report = engine.rag_pipeline.ingest_paths(path_list)
                st.write(report)
            if url_list:
                report = engine.rag_pipeline.ingest_urls(url_list)
                st.write(report)

    elif page == "Reports":
        st.subheader("Recent Reports")
        runs = engine.sqlite_store.list_recent_runs(limit=20)
        st.dataframe(runs)

    elif page == "Analytics":
        st.subheader("Analytics")
        st.plotly_chart(analytics.confidence_trend(), use_container_width=True)
        st.plotly_chart(analytics.status_distribution(), use_container_width=True)

    elif page == "Configuration":
        st.subheader("Loaded Configuration")
        st.code(json.dumps(config.model_dump(mode="json"), indent=2), language="json")

    engine.close()


if __name__ == "__main__":
    launch_dashboard()
