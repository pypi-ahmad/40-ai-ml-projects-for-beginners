"""Candidate summary and hiring recommendation generation."""

from __future__ import annotations

from resume_ai.models import HiringRecommendation, ResumeParseResult, ScoreBreakdown


def build_recommendation(parsed: ResumeParseResult, score: ScoreBreakdown) -> HiringRecommendation:
    """Generate explainable recommendation payload."""
    strengths = score.matched_skills[:6]
    weaknesses = score.missing_skills[:6]

    if score.total_score >= 80:
        recommendation = "strong_hire"
    elif score.total_score >= 65:
        recommendation = "hire"
    elif score.total_score >= 50:
        recommendation = "hold"
    else:
        recommendation = "reject"

    return HiringRecommendation(
        candidate_id=score.candidate_id,
        summary=parsed.summary,
        strengths=strengths,
        weaknesses=weaknesses,
        growth_potential="High" if len(weaknesses) <= 3 else "Moderate",
        career_progression="Steady based on role transitions listed in experience.",
        recommendation=recommendation,
        confidence_score=score.confidence,
    )
