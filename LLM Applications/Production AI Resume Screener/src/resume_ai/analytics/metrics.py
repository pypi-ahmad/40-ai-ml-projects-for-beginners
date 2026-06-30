"""Analytics aggregation and Plotly chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from resume_ai.db import models
from resume_ai.models import AnalyticsSnapshot


def build_snapshot(session: Session) -> AnalyticsSnapshot:
    scores = session.execute(select(models.CandidateJobScore.total_score)).scalars().all()
    avg = sum(scores) / len(scores) if scores else 0.0

    skills = session.execute(select(models.CandidateSkill.skill_name)).scalars().all()
    counts: dict[str, int] = {}
    for skill in skills:
        counts[skill] = counts.get(skill, 0) + 1

    total_candidates = session.scalar(select(func.count(models.Candidate.id))) or 0
    top_skills = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20])
    return AnalyticsSnapshot(top_skills=top_skills, avg_match_score=avg, total_candidates=total_candidates)


def top_skills_chart(snapshot: AnalyticsSnapshot):
    frame = pd.DataFrame(
        {
            "skill": list(snapshot.top_skills.keys()),
            "count": list(snapshot.top_skills.values()),
        }
    )
    return px.bar(frame, x="skill", y="count", title="Top Skills")


def score_distribution_chart(scores: list[float]):
    frame = pd.DataFrame({"score": scores})
    return px.histogram(frame, x="score", nbins=20, title="Candidate Match Score Distribution")
