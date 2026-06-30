"""Streamlit AI Analyst Dashboard."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_spreadsheet_analytics.analytics import AnalyticsEngine
from ai_spreadsheet_analytics.chat import ConversationalAnalyticsAssistant
from ai_spreadsheet_analytics.cleaning import DataCleaner
from ai_spreadsheet_analytics.config import Settings
from ai_spreadsheet_analytics.connectors.auth import build_service_account_client
from ai_spreadsheet_analytics.connectors.google_sheets import GoogleSheetsLoader, SheetLoadRequest
from ai_spreadsheet_analytics.insights import InsightEngine
from ai_spreadsheet_analytics.llm.ollama_rest import OllamaRESTClient
from ai_spreadsheet_analytics.quality import DataQualityProfiler
from ai_spreadsheet_analytics.reporting import ReportGenerator
from ai_spreadsheet_analytics.schemas import CleaningStrategy
from ai_spreadsheet_analytics.state_store import SQLiteStateStore
from ai_spreadsheet_analytics.visualization import VisualizationEngine


@st.cache_resource
def bootstrap() -> tuple[Settings, SQLiteStateStore, GoogleSheetsLoader]:
    """Initialize app services."""
    load_dotenv()
    settings = Settings()
    settings.ensure_directories()
    state = SQLiteStateStore(settings.state_db_path)
    client = build_service_account_client(settings.google_service_account_json, settings.scopes)
    loader = GoogleSheetsLoader(client=client, cache_dir=settings.cache_dir, state_store=state)
    return settings, state, loader


def main() -> None:
    """Run Streamlit app."""
    st.set_page_config(page_title="AI Spreadsheet Analytics", page_icon="📊", layout="wide")
    st.title("AI Spreadsheet Analytics Platform")
    st.caption("Google Sheets + deterministic Python analytics + local Ollama insights")

    try:
        settings, state_store, loader = bootstrap()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Initialization failed: {exc}")
        st.stop()

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    analytics = AnalyticsEngine()
    quality = DataQualityProfiler()
    cleaner = DataCleaner()
    viz = VisualizationEngine()
    llm = InsightEngine(OllamaRESTClient(settings.ollama_base_url), default_temperature=0.0)
    chat_assistant = ConversationalAnalyticsAssistant(analytics, llm, state_store, settings.ollama_primary_model)
    reporter = ReportGenerator(settings.report_dir)

    st.sidebar.header("Setup")
    st.sidebar.markdown(
        """
        1. Create Google Cloud project.
        2. Enable Google Sheets + Drive API.
        3. Create service account and JSON key.
        4. Share spreadsheet with service-account email.
        5. Set `GOOGLE_SERVICE_ACCOUNT_JSON` in `.env`.
        """
    )
    spreadsheet_ids = st.sidebar.text_area(
        "Spreadsheet IDs (one per line)",
        value="\n".join(settings.spreadsheet_ids),
        height=120,
    )
    worksheet_names = st.sidebar.text_input("Worksheet names (comma-separated, optional)", ",".join(settings.worksheet_names))

    sheet_ids = [line.strip() for line in spreadsheet_ids.splitlines() if line.strip()]
    worksheets = [item.strip() for item in worksheet_names.split(",") if item.strip()]

    if st.sidebar.button("Load Data"):
        if not sheet_ids:
            st.sidebar.warning("Provide at least one spreadsheet ID")
        else:
            requests: list[SheetLoadRequest] = []
            for sid in sheet_ids:
                if worksheets:
                    for ws in worksheets:
                        requests.append(SheetLoadRequest(spreadsheet_id=sid, worksheet_title=ws))
                else:
                    for meta in loader.inspect_spreadsheet(sid):
                        requests.append(
                            SheetLoadRequest(spreadsheet_id=sid, worksheet_title=str(meta["worksheet_title"]))
                        )

            bundle = loader.load_batch(requests=requests, use_cache=True)
            st.session_state["bundle"] = bundle
            st.session_state["df"] = bundle.combined()

    df: pd.DataFrame | None = st.session_state.get("df")
    if df is None or df.empty:
        st.info("Load spreadsheet data to start analytics.")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Dataset Health"):
        report = quality.profile("combined", df)
        st.json(report.metrics)
        for issue in report.issues:
            st.warning(f"{issue.check_name}: {issue.message}")

    st.subheader("Cleaning Options")
    missing_strategy = st.selectbox("Missing value strategy", ["median", "mean", "mode", "drop_rows", "drop_columns", "zero"], index=0)
    if st.button("Run Cleaning"):
        strategy = CleaningStrategy(missing_value_strategy=missing_strategy)
        cleaning_result = cleaner.clean("combined", df, strategy)
        st.session_state["clean_df"] = cleaning_result.cleaned
        st.success("Cleaning complete")
        for action in cleaning_result.actions:
            st.write(f"- {action}")

    clean_df: pd.DataFrame = st.session_state.get("clean_df", df)

    st.subheader("Analytics Dashboard")
    eda = analytics.run_full_eda(clean_df)
    st.json(eda["kpis"])
    for line in eda["kpi_narratives"]:
        st.write(f"- {line}")

    numeric_cols = list(clean_df.select_dtypes(include=["number"]).columns)
    categorical_cols = list(clean_df.select_dtypes(exclude=["number"]).columns)

    if numeric_cols:
        hist_col = st.selectbox("Histogram column", numeric_cols)
        st.plotly_chart(viz.histogram(clean_df, hist_col, f"Distribution: {hist_col}"), use_container_width=True)

    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Scatter X", numeric_cols, key="scatter_x")
        y_axis = st.selectbox("Scatter Y", numeric_cols, index=1, key="scatter_y")
        color = st.selectbox("Scatter Color", [""] + categorical_cols, key="scatter_color")
        st.plotly_chart(
            viz.scatter(clean_df, x=x_axis, y=y_axis, color=color or None, title="Scatter analysis"),
            use_container_width=True,
        )

    st.subheader("LLM Insights")
    role = st.selectbox(
        "Insight role",
        [
            "executive",
            "sales_analyst",
            "finance_analyst",
            "marketing_analyst",
            "operations_analyst",
            "product_manager",
            "data_scientist",
        ],
    )

    if st.button("Generate Insights"):
        try:
            packet = llm.generate(eda, role=role, model=settings.ollama_primary_model)
            st.markdown(packet.summary)
            st.caption(
                f"Model: {packet.model} | Latency: {packet.latency_ms:.1f} ms | Tokens~{packet.token_estimate}"
            )
            st.session_state["latest_insight"] = packet.summary
        except Exception as exc:  # noqa: BLE001
            st.error(f"LLM request failed: {exc}")

    st.subheader("Conversational Analytics")
    user_q = st.chat_input("Ask: Which products sold best? Which month had highest revenue?")
    if user_q:
        st.chat_message("user").write(user_q)
        try:
            turn = chat_assistant.ask(user_q, clean_df, st.session_state["session_id"], role=role)
            st.chat_message("assistant").write(turn.answer)
        except Exception as exc:  # noqa: BLE001
            st.chat_message("assistant").write(
                "LLM response unavailable. Deterministic analytics succeeded but narrative generation failed."
            )
            st.error(str(exc))

    st.subheader("Download Reports")
    if st.button("Generate Report Pack"):
        insight_md = st.session_state.get("latest_insight", "No LLM insight generated yet.")
        tables = {
            "cleaned_preview": clean_df.head(100),
            "numeric_summary": pd.DataFrame(eda["summary"]["numeric_summary"]),
        }
        artifacts = reporter.generate("AI Analyst Report", insight_md, tables)
        st.success("Reports generated")
        st.json({k: str(v) for k, v in artifacts.model_dump().items() if v})

    st.subheader("AI Analyst Dashboard")
    history = state_store.get_chat_history(st.session_state["session_id"], limit=20)
    st.metric("Rows", len(clean_df))
    st.metric("Columns", len(clean_df.columns))
    st.metric("Chat Queries", len(history))


if __name__ == "__main__":
    main()
