"""Streamlit recruiter dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from resume_ai.analytics.metrics import score_distribution_chart, top_skills_chart
from resume_ai.models import AnalyticsSnapshot
from resume_ai.service import ResumeAIService

service = ResumeAIService()

st.set_page_config(page_title="AI Resume Screener", layout="wide")

PAGES = [
    "Dashboard",
    "Resume Upload",
    "Candidate Explorer",
    "Job Description",
    "Ranking",
    "Comparison",
    "Interview Generator",
    "Analytics",
    "Reports",
    "Settings",
]


def _dashboard() -> None:
    st.title("Production AI Resume Screening & Talent Intelligence")
    analytics = service.analytics()
    snapshot = analytics["snapshot"]
    st.metric("Total Candidates", snapshot["total_candidates"])
    st.metric("Average Match Score", round(snapshot["avg_match_score"], 2))


def _upload() -> None:
    st.title("Resume Upload")
    blind = st.toggle("Blind hiring mode", value=True)
    path = st.text_input("Resume file path")
    if st.button("Upload Resume") and path:
        with st.spinner("Processing..."):
            out = service.upload_resume(path, blind_mode=blind)
        st.json(out)

    folder = st.text_input("Batch folder path")
    if st.button("Batch Ingest") and folder:
        queued = service.enqueue_folder(folder)
        ran = service.run_queue(blind_mode=blind)
        st.json({**queued, **ran})


def _candidate_explorer() -> None:
    st.title("Candidate Explorer")
    candidate_id = st.number_input("Candidate ID", min_value=1, value=1)
    if st.button("Load Candidate"):
        try:
            out = service.get_candidate(int(candidate_id))
            st.json(out)
        except Exception as exc:
            st.error(str(exc))

    note = st.text_area("Recruiter note")
    if st.button("Save Note") and note:
        note_id = service.add_note(int(candidate_id), note)
        st.success(f"Saved note #{note_id}")


def _job_page() -> None:
    st.title("Job Description")
    jd_text = st.text_area("Paste JD", height=280)
    if st.button("Parse JD") and jd_text:
        out = service.create_job(jd_text)
        st.json(out)


def _ranking() -> None:
    st.title("Ranking")
    job_id = st.number_input("Job ID", min_value=1, value=1, key="rank_job")
    candidate_id = st.number_input("Candidate ID", min_value=1, value=1, key="rank_candidate")
    if st.button("Score Candidate"):
        out = service.score(int(candidate_id), int(job_id))
        st.json(out.model_dump(mode="json"))

    if st.button("Load Ranked List"):
        ranked = service.rank_for_job(int(job_id))
        st.dataframe(pd.DataFrame(ranked))


def _comparison() -> None:
    st.title("Candidate Comparison")
    job_id = st.number_input("Job ID", min_value=1, value=1, key="cmp_job")
    raw = st.text_input("Candidate IDs (comma-separated)", value="1,2")
    if st.button("Compare"):
        ids = [int(part.strip()) for part in raw.split(",") if part.strip().isdigit()]
        out = service.compare(int(job_id), ids)
        st.dataframe(pd.DataFrame(out))


def _interview() -> None:
    st.title("Interview Generator")
    candidate_id = st.number_input("Candidate ID", min_value=1, value=1, key="int_candidate")
    job_id = st.number_input("Job ID", min_value=1, value=1, key="int_job")
    if st.button("Generate Interview Pack"):
        out = service.generate_interview(int(candidate_id), int(job_id))
        st.json(out)


def _analytics() -> None:
    st.title("Analytics")
    payload = service.analytics()
    snapshot = AnalyticsSnapshot.model_validate(payload["snapshot"])
    chart1 = top_skills_chart(snapshot)
    chart2 = score_distribution_chart(payload["scores"])
    st.plotly_chart(chart1, use_container_width=True)
    st.plotly_chart(chart2, use_container_width=True)


def _reports() -> None:
    st.title("Reports")
    candidate_id = st.number_input("Candidate ID", min_value=1, value=1, key="rep_candidate")
    job_id = st.number_input("Job ID", min_value=1, value=1, key="rep_job")
    if st.button("Generate Report"):
        out = service.generate_report(int(candidate_id), int(job_id))
        st.json(out)

    st.subheader("Existing Reports")
    st.dataframe(pd.DataFrame(service.list_reports()))


def _settings() -> None:
    st.title("Settings")
    st.code(json.dumps(service.config.model_dump(mode="json"), indent=2), language="json")


def run() -> None:
    page = st.sidebar.radio("Pages", PAGES)
    page_map = {
        "Dashboard": _dashboard,
        "Resume Upload": _upload,
        "Candidate Explorer": _candidate_explorer,
        "Job Description": _job_page,
        "Ranking": _ranking,
        "Comparison": _comparison,
        "Interview Generator": _interview,
        "Analytics": _analytics,
        "Reports": _reports,
        "Settings": _settings,
    }
    page_map[page]()


if __name__ == "__main__":
    run()
