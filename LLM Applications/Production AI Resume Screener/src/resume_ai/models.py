"""Pydantic schemas for resume intelligence platform."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class OCRMode(str, Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"


class CandidateContact(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    portfolio: str | None = None


class EducationItem(BaseModel):
    degree: str
    university: str | None = None
    cgpa: str | None = None
    graduation_year: int | None = None


class ExperienceItem(BaseModel):
    company: str
    role: str
    duration: str | None = None
    responsibilities: list[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    title: str
    technologies: list[str] = Field(default_factory=list)
    impact: str | None = None


class SkillSet(BaseModel):
    technical: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    ai_ml: list[str] = Field(default_factory=list)


class ResumeParseResult(BaseModel):
    candidate: CandidateContact = Field(default_factory=CandidateContact)
    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    skills: SkillSet = Field(default_factory=SkillSet)
    certifications: list[str] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    summary: str = ""
    ocr_mode: OCRMode = OCRMode.DIGITAL
    redacted_text: str | None = None


class JobRequirementProfile(BaseModel):
    title: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_requirements: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)


class ScoreEvidence(BaseModel):
    source: str
    snippet: str


class ScoreBreakdown(BaseModel):
    candidate_id: int | None = None
    job_id: int | None = None
    total_score: float = 0.0
    technical_skills: float = 0.0
    experience: float = 0.0
    projects: float = 0.0
    education: float = 0.0
    certifications: float = 0.0
    communication: float = 0.0
    bonus_skills: float = 0.0
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    evidence: list[ScoreEvidence] = Field(default_factory=list)
    confidence: float = 0.0
    weight_profile: dict[str, float] = Field(default_factory=dict)


class InterviewDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class InterviewQuestion(BaseModel):
    category: str
    difficulty: InterviewDifficulty
    question: str
    follow_up: str | None = None


class InterviewQuestionSet(BaseModel):
    candidate_id: int | None = None
    job_id: int | None = None
    technical: list[InterviewQuestion] = Field(default_factory=list)
    behavioral: list[InterviewQuestion] = Field(default_factory=list)
    coding: list[InterviewQuestion] = Field(default_factory=list)
    project: list[InterviewQuestion] = Field(default_factory=list)


class SkillGapReport(BaseModel):
    candidate_id: int | None = None
    job_id: int | None = None
    missing_skills: list[str] = Field(default_factory=list)
    recommended_learning: list[str] = Field(default_factory=list)
    certification_suggestions: list[str] = Field(default_factory=list)
    estimated_readiness: float = 0.0


class HiringRecommendation(BaseModel):
    candidate_id: int | None = None
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    growth_potential: str
    career_progression: str
    recommendation: str
    confidence_score: float


class CandidateSearchResult(BaseModel):
    candidate_id: int
    name: str
    score: float
    highlights: list[str] = Field(default_factory=list)


class AnalyticsSnapshot(BaseModel):
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    top_skills: dict[str, int] = Field(default_factory=dict)
    avg_match_score: float = 0.0
    total_candidates: int = 0


class APIEnvelope(BaseModel):
    status: str = "ok"
    data: dict
    errors: list[str] = Field(default_factory=list)
    trace_id: str | None = None
