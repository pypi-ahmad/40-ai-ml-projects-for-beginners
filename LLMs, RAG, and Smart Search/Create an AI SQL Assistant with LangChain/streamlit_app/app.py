"""Streamlit UI for AI SQL Analytics Assistant."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_sql_assistant.config import get_settings
from ai_sql_assistant.constants import APP_NAME, APP_VERSION
from ai_sql_assistant.data.northwind import build_northwind_databases
from ai_sql_assistant.pipeline.assistant import AISQLAssistant
from ai_sql_assistant.types import QueryRequest, VisualizationSpec
from ai_sql_assistant.visualization.recommender import render_chart

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_assistant() -> AISQLAssistant:
    settings = get_settings()
    # Ensure databases exist before constructing assistant.
    if not settings.database.scaled_db_path.exists() or not settings.database.raw_db_path.exists():
        build_northwind_databases(
            raw_db_path=settings.database.raw_db_path,
            scaled_db_path=settings.database.scaled_db_path,
            scale_factor=8,
        )
    return AISQLAssistant(settings)


def _download_bytes(dataframe: pd.DataFrame, fmt: str) -> bytes:
    if fmt == "csv":
        return dataframe.to_csv(index=False).encode("utf-8")
    buffer = io.BytesIO()
    dataframe.to_excel(buffer, index=False)
    return buffer.getvalue()


def _render_sidebar() -> dict[str, str]:
    st.sidebar.title("Assistant Controls")
    persona = st.sidebar.selectbox(
        "Persona",
        ["Business Analyst", "Finance", "Sales", "HR", "Inventory", "Marketing"],
        index=0,
    )
    approach = st.sidebar.selectbox("Generation Approach", ["langchain", "direct"], index=0)
    model = st.sidebar.selectbox("Model", ["qwen3.5:4b", "granite4.1:3b"], index=0)

    conversation_id = st.sidebar.text_input("Conversation ID", value="streamlit-session")
    st.sidebar.caption("Deterministic decoding enabled (`temperature=0`, fixed seed).")

    return {
        "persona": persona,
        "approach": approach,
        "model": model,
        "conversation_id": conversation_id,
    }


def _ask_tab(assistant: AISQLAssistant, controls: dict[str, str]) -> None:
    st.subheader("Natural Language to SQL")
    question = st.text_area(
        "Ask business question",
        value="Show monthly net revenue for Europe in 2024.",
        height=100,
    )

    if st.button("Generate + Validate + Execute", type="primary", use_container_width=True):
        req = QueryRequest(
            question=question,
            persona=controls["persona"],
            conversation_id=controls["conversation_id"],
            user_id="streamlit",
        )
        with st.spinner("Running assistant pipeline..."):
            response = assistant.ask(req, approach=controls["approach"], model=controls["model"])

        st.session_state["last_response"] = response

    response = st.session_state.get("last_response")
    if response is None:
        st.info("Run query to see SQL, validation, explanation, and visualizations.")
        return

    st.markdown("### Generated SQL")
    st.code(response.execution.sql, language="sql")

    if response.validation.issues:
        st.markdown("### Validation")
        for issue in response.validation.issues:
            severity = "❌" if issue.severity == "error" else "⚠️"
            st.write(f"{severity} `{issue.code}`: {issue.message}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", response.execution.status)
    col2.metric("Rows", response.execution.row_count)
    col3.metric("Execution ms", f"{response.execution.execution_time_ms:.2f}")
    col4.metric("Complexity", f"{response.execution.complexity_score:.2f}")

    st.markdown("### SQL Explanation")
    st.write(response.explanation)

    if response.execution.rows:
        st.markdown("### Results")
        frame = pd.DataFrame(response.execution.rows)
        st.dataframe(frame, use_container_width=True)

        options = response.visualization_options
        chart_types = [item.chart_type for item in options]
        selected = st.selectbox("Chart Type", chart_types, index=0)

        selected_spec = next((item for item in options if item.chart_type == selected), VisualizationSpec(chart_type="table"))
        fig = render_chart(response.execution.rows, selected_spec)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.download_button(
            "Download CSV",
            data=_download_bytes(frame, "csv"),
            file_name=f"query_result_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        c2.download_button(
            "Download Excel",
            data=_download_bytes(frame, "xlsx"),
            file_name=f"query_result_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        c3.download_button(
            "Download SQL",
            data=response.execution.sql.encode("utf-8"),
            file_name=f"query_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.sql",
            mime="text/plain",
        )

        if c4.button("Save Favorite"):
            assistant.add_favorite(label=question[:60], question=question, sql=response.execution.sql)
            st.success("Saved to favorites")


def _schema_tab(assistant: AISQLAssistant) -> None:
    st.subheader("Schema Browser")
    payload = assistant.schema_browser_payload()
    report = payload["schema_report"]
    summary = payload["schema_summary"]

    st.write(f"Active DB: `{report['database']}`")
    table_names = list(report["tables"].keys())
    selected_table = st.selectbox("Select table", table_names)

    table_meta = report["tables"][selected_table]
    st.metric("Row Count", table_meta["row_count"])

    st.markdown("#### Columns")
    st.dataframe(pd.DataFrame(table_meta["columns"]), use_container_width=True)

    st.markdown("#### Foreign Keys")
    st.dataframe(pd.DataFrame(table_meta["foreign_keys"]), use_container_width=True)

    st.markdown("#### Indexes")
    st.dataframe(pd.DataFrame(table_meta["indexes"]), use_container_width=True)

    st.markdown("#### Null Statistics")
    null_frame = pd.DataFrame(list(table_meta["null_stats"].items()), columns=["column", "null_count"])
    st.dataframe(null_frame, use_container_width=True)

    st.markdown("#### Sample Rows")
    st.dataframe(pd.DataFrame(table_meta["sample_rows"]), use_container_width=True)

    st.markdown("#### Business Summary")
    st.json(summary["tables"].get(selected_table, {}))


def _history_tab(assistant: AISQLAssistant) -> None:
    st.subheader("Query History & Dashboard")
    stats = assistant.dashboard_stats()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Queries", stats.get("total_queries", 0))
    m2.metric("Success Rate", f"{stats.get('success_rate', 0.0) * 100:.1f}%")
    m3.metric("Avg Latency (ms)", f"{stats.get('avg_latency_ms', 0.0):.1f}")

    top_tables = stats.get("top_tables", [])
    st.write("Frequent tables:", ", ".join(top_tables) if top_tables else "N/A")

    history = assistant.history_frame(limit=200)
    favorites = assistant.favorites_frame()

    st.markdown("### Recent Query History")
    st.dataframe(history, use_container_width=True)

    st.markdown("### Favorites")
    st.dataframe(favorites, use_container_width=True)


def _about_tab() -> None:
    st.subheader("Why deterministic decoding (`temperature=0`) for SQL")
    st.markdown(
        """
- Reproducible SQL for validation and benchmark regression.
- Lower random syntax drift means fewer execution failures.
- Stable outputs improve safety auditing and governance.
- Determinism simplifies debugging and model comparison.
        """
    )


def main() -> None:
    st.title(f"{APP_NAME} v{APP_VERSION}")
    controls = _render_sidebar()
    assistant = get_assistant()

    tab_ask, tab_schema, tab_history, tab_about = st.tabs(
        ["Ask", "Schema Browser", "History Dashboard", "About"]
    )

    with tab_ask:
        _ask_tab(assistant, controls)

    with tab_schema:
        _schema_tab(assistant)

    with tab_history:
        _history_tab(assistant)

    with tab_about:
        _about_tab()


if __name__ == "__main__":
    main()
