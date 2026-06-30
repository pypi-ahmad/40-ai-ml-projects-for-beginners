"""Streamlit dashboard for planning agent."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from task_planning_agent.agent.service import PlanningService
from task_planning_agent.calendar.service import CalendarService
from task_planning_agent.config import load_config
from task_planning_agent.schemas import PriorityStrategy


def _service() -> PlanningService:
    return PlanningService(load_config())


def _calendar() -> CalendarService:
    return CalendarService()


def _style() -> None:
    st.set_page_config(page_title="AI Task Productivity Agent", page_icon="🧠", layout="wide")
    st.markdown(
        """
        <style>
        :root {
          --bg: #0f172a;
          --card: #111827;
          --text: #f8fafc;
          --accent: #14b8a6;
          --accent2: #f59e0b;
        }
        .stApp { background: radial-gradient(circle at 10% 10%, #1f2937 0%, #0f172a 55%); color: var(--text); }
        .metric-card { background: var(--card); border-radius: 14px; padding: 12px; border: 1px solid #334155; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plan_to_df(report) -> pd.DataFrame:
    rows = []
    for item in report.schedule:
        rows.append(
            {
                "Task": item.task,
                "Priority": item.priority,
                "Deadline": item.deadline,
                "Start": item.suggested_start_time,
                "End": item.suggested_end_time,
                "Risk": item.risk_level.value,
                "Confidence": item.confidence,
            }
        )
    return pd.DataFrame(rows)


def dashboard_page() -> None:
    st.title("Dashboard")
    user_id = st.text_input("User ID", value="ahmad")
    strategy = st.selectbox("Priority strategy", [p.value for p in PriorityStrategy], index=5)
    raw = st.text_area(
        "Task Inbox",
        value="""
- Finish proposal deck by tomorrow 5pm #client @sara 120min
- Review pull requests before today 6pm 45min
- Prepare sprint planning notes for next monday 90min
- Weekly analytics sync meeting friday 11am 60min
""".strip(),
        height=180,
    )

    if st.button("Generate Plan", type="primary"):
        report = _service().plan(
            user_id=user_id,
            raw_input=raw,
            strategy=PriorityStrategy(strategy),
            timezone="Asia/Kolkata",
        )
        st.session_state["latest_report"] = report

    report = st.session_state.get("latest_report")
    if report is None:
        st.info("Generate plan to view dashboard.")
        return

    df = _plan_to_df(report)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasks", len(df))
    c2.metric("Completion Rate", f"{report.analytics.completion_rate * 100:.1f}%")
    c3.metric("Focus Minutes", report.analytics.focus_time_minutes)
    c4.metric("Burnout Score", f"{report.analytics.burnout_score:.1f}")

    st.subheader("Timeline")
    if not df.empty:
        fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Risk")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)


def today_page() -> None:
    st.title("Today's Plan")
    report = st.session_state.get("latest_report")
    if report is None:
        st.warning("No plan generated yet.")
        return
    df = _plan_to_df(report)
    st.data_editor(df, use_container_width=True, num_rows="dynamic")


def weekly_page() -> None:
    st.title("Weekly Planner")
    report = st.session_state.get("latest_report")
    if report is None:
        st.warning("No plan generated yet.")
        return
    df = _plan_to_df(report)
    st.subheader("Gantt")
    if not df.empty:
        fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Priority")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Priority Matrix")
    if not df.empty:
        fig2 = px.scatter(df, x="Confidence", y="Priority", size="Priority", color="Risk", hover_name="Task")
        st.plotly_chart(fig2, use_container_width=True)


def inbox_page() -> None:
    st.title("Task Inbox")
    user_id = st.text_input("User ID", value="ahmad", key="inbox_user")
    query = st.text_input("Search tasks")
    if st.button("Load Tasks"):
        st.session_state["tasks_view"] = _service().memory.sqlite.list_tasks(user_id)
    tasks = st.session_state.get("tasks_view", [])
    rows = [task.model_dump() for task in tasks]
    if query:
        rows = [row for row in rows if query.lower() in str(row).lower()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def calendar_page() -> None:
    st.title("Calendar")
    cal = _calendar()
    report = st.session_state.get("latest_report")
    path = st.text_input("Export ICS Path", value="artifacts/reports/latest_schedule.ics")
    if st.button("Export current plan to ICS") and report is not None:
        session = _service().memory.history(user_id=report.user_id, limit=1)
        if session:
            output = cal.export_ics(path, session[0].schedule_blocks)
            st.success(f"Exported: {output}")

    upload = st.file_uploader("Import ICS", type=["ics"])
    if upload is not None:
        temp_path = "artifacts/reports/uploaded.ics"
        with open(temp_path, "wb") as handle:
            handle.write(upload.read())
        events = cal.import_ics(temp_path)
        st.dataframe(pd.DataFrame(events), use_container_width=True)


def memory_page() -> None:
    st.title("Memory")
    user_id = st.text_input("User ID", value="ahmad", key="memory_user")
    query = st.text_input("Semantic Search", value="deadline risk")
    history = _service().memory.history(user_id=user_id, limit=5)
    st.caption(f"Stored plan sessions for user: {len(history)}")
    if st.button("Search Memory"):
        st.session_state["memory_results"] = _service().memory.semantic_search(query)
    results = st.session_state.get("memory_results", [])
    st.json(results)


def analytics_page() -> None:
    st.title("Analytics")
    report = st.session_state.get("latest_report")
    if report is None:
        st.info("No analytics yet.")
        return
    metrics = report.analytics.model_dump()
    df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items() if isinstance(v, (int, float))])
    fig = px.bar(df, x="metric", y="value", title="Productivity Metrics")
    st.plotly_chart(fig, use_container_width=True)


def reports_page() -> None:
    st.title("Reports")
    report = st.session_state.get("latest_report")
    if report is None:
        st.info("No report available.")
        return
    st.json(report.model_dump(mode="json"))


def settings_page() -> None:
    st.title("Settings")
    cfg = load_config().raw
    st.json(cfg)


def render() -> None:
    _style()
    st.sidebar.title("Navigation")
    pages = {
        "Dashboard": dashboard_page,
        "Today's Plan": today_page,
        "Weekly Planner": weekly_page,
        "Task Inbox": inbox_page,
        "Calendar": calendar_page,
        "Memory": memory_page,
        "Analytics": analytics_page,
        "Reports": reports_page,
        "Settings": settings_page,
    }
    page = st.sidebar.radio("Page", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    render()
