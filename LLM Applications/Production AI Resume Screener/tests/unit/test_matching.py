from resume_ai.config.loader import AppConfig
from resume_ai.matching.engine import MatchingEngine
from resume_ai.models import JobRequirementProfile, ResumeParseResult


def test_matching_scores_and_missing_skills() -> None:
    cfg = AppConfig()
    engine = MatchingEngine(cfg)

    class DummyEmbedding:
        def embed_text(self, _text: str):
            return [1.0, 0.0]

    engine.embedding = DummyEmbedding()  # type: ignore[assignment]

    resume = ResumeParseResult.model_validate(
        {
            "candidate": {"name": "A"},
            "education": [{"degree": "Bachelor"}],
            "experience": [{"company": "X", "role": "ML Engineer", "responsibilities": ["Built APIs"]}],
            "projects": [{"title": "RAG", "technologies": ["langchain", "fastapi"]}],
            "skills": {
                "technical": ["python", "fastapi", "langchain"],
                "soft": ["communication"],
                "cloud": ["aws"],
                "programming_languages": ["python"],
                "frameworks": ["fastapi"],
                "databases": ["sqlite"],
                "ai_ml": ["langchain"],
            },
            "certifications": ["AWS"],
            "awards": [],
            "publications": [],
            "languages": [],
            "summary": "Built production RAG systems",
            "ocr_mode": "digital",
        }
    )

    jd = JobRequirementProfile.model_validate(
        {
            "required_skills": ["python", "kubernetes"],
            "preferred_skills": ["langchain"],
            "technologies": ["python", "langchain", "kubernetes"],
            "experience_requirements": ["3 years"],
            "education_requirements": ["bachelor"],
            "responsibilities": ["Build APIs"],
            "keywords": ["python", "kubernetes", "rag"],
            "soft_skills": ["communication"],
        }
    )

    out = engine.score_candidate(resume, jd, candidate_id=1, job_id=1)

    assert out.total_score >= 0
    assert "python" in out.matched_skills
    assert "kubernetes" in out.missing_skills
