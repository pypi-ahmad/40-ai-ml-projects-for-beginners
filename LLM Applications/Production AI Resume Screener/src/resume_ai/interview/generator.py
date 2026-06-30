"""Interview question generation from score gaps."""

from __future__ import annotations

from resume_ai.config.loader import AppConfig
from resume_ai.models import InterviewDifficulty, InterviewQuestion, InterviewQuestionSet, ScoreBreakdown


class InterviewGenerator:
    """Generate technical/behavioral/coding/project interview packs."""

    def __init__(self, _config: AppConfig):
        self.config = _config

    def generate(self, score: ScoreBreakdown) -> InterviewQuestionSet:
        missing = score.missing_skills[:5]
        technical = [
            InterviewQuestion(
                category="technical",
                difficulty=InterviewDifficulty.MEDIUM,
                question=f"Explain your hands-on experience with {skill} in production systems.",
                follow_up=f"What tradeoff did you make while implementing {skill}?",
            )
            for skill in missing
        ]

        behavioral = [
            InterviewQuestion(
                category="behavioral",
                difficulty=InterviewDifficulty.MEDIUM,
                question="Describe conflict in cross-functional team and how you resolved it.",
                follow_up="What did you change afterward in your collaboration style?",
            )
        ]

        coding = [
            InterviewQuestion(
                category="coding",
                difficulty=InterviewDifficulty.HARD,
                question="Design scalable resume ranking service with explainable scoring API.",
                follow_up="How would you test ranking fairness and regression drift?",
            )
        ]

        project = [
            InterviewQuestion(
                category="project",
                difficulty=InterviewDifficulty.MEDIUM,
                question="Pick one project from your resume and explain architecture decisions.",
                follow_up="What monitoring did you add after production launch?",
            )
        ]

        return InterviewQuestionSet(
            candidate_id=score.candidate_id,
            job_id=score.job_id,
            technical=technical,
            behavioral=behavioral,
            coding=coding,
            project=project,
        )
