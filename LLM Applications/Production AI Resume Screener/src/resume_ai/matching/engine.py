"""Semantic and rule-based candidate scoring engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from resume_ai.config.loader import AppConfig, ScoringConfig
from resume_ai.embeddings.service import EmbeddingService, cosine_similarity
from resume_ai.models import JobRequirementProfile, ResumeParseResult, ScoreBreakdown, ScoreEvidence


def _safe_ratio(hit: int, total: int) -> float:
    return 0.0 if total == 0 else min(1.0, hit / total)


def _flatten_resume_skills(parsed: ResumeParseResult) -> set[str]:
    skills = set(parsed.skills.technical)
    skills.update(parsed.skills.programming_languages)
    skills.update(parsed.skills.frameworks)
    skills.update(parsed.skills.databases)
    skills.update(parsed.skills.cloud)
    skills.update(parsed.skills.ai_ml)
    return {item.lower().strip() for item in skills if item}


@dataclass(slots=True)
class MatchingEngine:
    config: AppConfig
    embedding: EmbeddingService = field(init=False)

    def __post_init__(self) -> None:
        self.embedding = EmbeddingService(self.config)

    def score_candidate(
        self,
        parsed_resume: ResumeParseResult,
        parsed_job: JobRequirementProfile,
        candidate_id: int | None = None,
        job_id: int | None = None,
        override_weights: dict[str, float] | None = None,
    ) -> ScoreBreakdown:
        weights = self._resolve_weights(override_weights)

        resume_skills = _flatten_resume_skills(parsed_resume)
        required = {skill.lower() for skill in parsed_job.required_skills}
        preferred = {skill.lower() for skill in parsed_job.preferred_skills}

        matched = sorted(resume_skills.intersection(required.union(preferred)))
        missing = sorted(required.difference(resume_skills))

        tech_score = _safe_ratio(len(resume_skills.intersection(required)), max(1, len(required)))
        exp_score = self._experience_score(parsed_resume, parsed_job)
        proj_score = self._project_score(parsed_resume, parsed_job)
        edu_score = self._education_score(parsed_resume, parsed_job)
        cert_score = self._certification_score(parsed_resume)
        comm_score = self._communication_score(parsed_resume)
        bonus_score = _safe_ratio(len(resume_skills.intersection(preferred)), max(1, len(preferred)))

        semantic_score = self._semantic_score(parsed_resume.summary, " ".join(parsed_job.keywords))
        tech_score = min(1.0, (tech_score + semantic_score) / 2)

        total = (
            tech_score * weights.technical_skills
            + exp_score * weights.experience
            + proj_score * weights.projects
            + edu_score * weights.education
            + cert_score * weights.certifications
            + comm_score * weights.communication
            + bonus_score * weights.bonus_skills
        )

        evidence = [
            ScoreEvidence(source="skills", snippet=", ".join(matched[:15])),
            ScoreEvidence(source="summary", snippet=parsed_resume.summary[:240]),
        ]

        return ScoreBreakdown(
            candidate_id=candidate_id,
            job_id=job_id,
            total_score=round(total * 100, 2),
            technical_skills=round(tech_score * 100, 2),
            experience=round(exp_score * 100, 2),
            projects=round(proj_score * 100, 2),
            education=round(edu_score * 100, 2),
            certifications=round(cert_score * 100, 2),
            communication=round(comm_score * 100, 2),
            bonus_skills=round(bonus_score * 100, 2),
            matched_skills=matched,
            missing_skills=missing,
            evidence=evidence,
            confidence=round((0.6 + 0.4 * semantic_score) * 100, 2),
            weight_profile=weights.model_dump(),
        )

    def _resolve_weights(self, override: dict[str, float] | None) -> ScoringConfig:
        if not override:
            return self.config.scoring
        data = self.config.scoring.model_dump()
        data.update(override)
        return ScoringConfig.model_validate(data)

    @staticmethod
    def _experience_score(parsed_resume: ResumeParseResult, parsed_job: JobRequirementProfile) -> float:
        candidate_year_signals = len(parsed_resume.experience)
        required_year_signals = len(parsed_job.experience_requirements)
        if required_year_signals == 0:
            return min(1.0, candidate_year_signals / 4)
        return _safe_ratio(candidate_year_signals, required_year_signals)

    @staticmethod
    def _project_score(parsed_resume: ResumeParseResult, parsed_job: JobRequirementProfile) -> float:
        if not parsed_resume.projects:
            return 0.0
        job_tech = {token.lower() for token in parsed_job.technologies}
        overlap = 0
        total = 0
        for project in parsed_resume.projects:
            tech = {t.lower() for t in project.technologies}
            if not tech:
                continue
            total += len(tech)
            overlap += len(tech.intersection(job_tech))
        if total == 0:
            return min(1.0, len(parsed_resume.projects) / 3)
        return overlap / total

    @staticmethod
    def _education_score(parsed_resume: ResumeParseResult, parsed_job: JobRequirementProfile) -> float:
        if not parsed_job.education_requirements:
            return 1.0 if parsed_resume.education else 0.5
        req = {item.lower() for item in parsed_job.education_requirements}
        candidate = {edu.degree.lower() for edu in parsed_resume.education}
        return _safe_ratio(len(candidate.intersection(req)), len(req))

    @staticmethod
    def _certification_score(parsed_resume: ResumeParseResult) -> float:
        return min(1.0, len(parsed_resume.certifications) / 3)

    @staticmethod
    def _communication_score(parsed_resume: ResumeParseResult) -> float:
        soft = {item.lower() for item in parsed_resume.skills.soft}
        return 1.0 if "communication" in soft else 0.5

    def _semantic_score(self, resume_summary: str, job_keywords: str) -> float:
        if not resume_summary or not job_keywords:
            return 0.0
        resume_emb = self.embedding.embed_text(resume_summary)
        jd_emb = self.embedding.embed_text(job_keywords)
        return max(0.0, cosine_similarity(resume_emb, jd_emb))
