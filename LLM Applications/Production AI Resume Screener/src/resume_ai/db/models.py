"""SQLAlchemy models for ATS-like workflow."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    linkedin: Mapped[str | None] = mapped_column(String(255), nullable=True)
    github: Mapped[str | None] = mapped_column(String(255), nullable=True)
    portfolio: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    resumes: Mapped[list[Resume]] = relationship(back_populates="candidate", cascade="all, delete-orphan")


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    file_name: Mapped[str] = mapped_column(String(255))
    file_hash: Mapped[str] = mapped_column(String(128), index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    parsed_json: Mapped[dict] = mapped_column(JSON)
    redacted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_mode: Mapped[str] = mapped_column(String(32), default="digital")
    ingestion_status: Mapped[str] = mapped_column(String(32), default="processed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    candidate: Mapped[Candidate] = relationship(back_populates="resumes")


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    jd_text: Mapped[str] = mapped_column(Text)
    parsed_json: Mapped[dict] = mapped_column(JSON)
    weight_override: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CandidateJobScore(Base):
    __tablename__ = "candidate_job_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("job_descriptions.id"), index=True)
    total_score: Mapped[float] = mapped_column(Float, default=0.0)
    breakdown_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CandidateSkill(Base):
    __tablename__ = "candidate_skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    skill_name: Mapped[str] = mapped_column(String(128), index=True)
    skill_type: Mapped[str] = mapped_column(String(64), default="technical")


class CandidateExperience(Base):
    __tablename__ = "experience"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    company: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(255))
    duration: Mapped[str | None] = mapped_column(String(128), nullable=True)
    responsibilities_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)


class CandidateProject(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    technologies_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    impact: Mapped[str | None] = mapped_column(Text, nullable=True)


class InterviewPack(Base):
    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("job_descriptions.id"), index=True)
    questions_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int | None] = mapped_column(ForeignKey("candidates.id"), nullable=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("job_descriptions.id"), nullable=True)
    report_type: Mapped[str] = mapped_column(String(64))
    artifact_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RecruiterNote(Base):
    __tablename__ = "recruiter_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    note: Mapped[str] = mapped_column(Text)
    tags_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    bookmarked: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(64), default="new")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_path: Mapped[str] = mapped_column(String(512), index=True)
    file_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class SystemMetric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    dimension_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


Index("idx_score_candidate_job", CandidateJobScore.candidate_id, CandidateJobScore.job_id)
Index("idx_resume_candidate_hash", Resume.candidate_id, Resume.file_hash)
