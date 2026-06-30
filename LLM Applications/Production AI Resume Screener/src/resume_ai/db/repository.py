"""Repository helpers for common DB operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from resume_ai.db import models
from resume_ai.models import JobRequirementProfile, ResumeParseResult, ScoreBreakdown


def upsert_candidate(session: Session, parsed: ResumeParseResult) -> models.Candidate:
    """Create or update candidate from parsed resume."""
    email = parsed.candidate.email
    candidate = None
    if email:
        candidate = session.scalar(select(models.Candidate).where(models.Candidate.email == str(email)))

    if candidate is None:
        candidate = models.Candidate(
            name=parsed.candidate.name,
            email=str(email) if email else None,
            phone=parsed.candidate.phone,
            location=parsed.candidate.location,
            linkedin=parsed.candidate.linkedin,
            github=parsed.candidate.github,
            portfolio=parsed.candidate.portfolio,
        )
        session.add(candidate)
        session.flush()
    else:
        candidate.name = parsed.candidate.name or candidate.name
        candidate.phone = parsed.candidate.phone or candidate.phone
        candidate.location = parsed.candidate.location or candidate.location
        candidate.linkedin = parsed.candidate.linkedin or candidate.linkedin
        candidate.github = parsed.candidate.github or candidate.github
        candidate.portfolio = parsed.candidate.portfolio or candidate.portfolio
    return candidate


def insert_resume(
    session: Session,
    candidate_id: int,
    file_name: str,
    file_hash: str,
    raw_text: str,
    parsed: ResumeParseResult,
) -> models.Resume:
    resume = models.Resume(
        candidate_id=candidate_id,
        file_name=file_name,
        file_hash=file_hash,
        raw_text=raw_text,
        parsed_json=parsed.model_dump(mode="json"),
        redacted_text=parsed.redacted_text,
        ocr_mode=parsed.ocr_mode.value,
    )
    session.add(resume)
    session.flush()
    return resume


def sync_candidate_details(session: Session, candidate_id: int, parsed: ResumeParseResult) -> None:
    """Persist denormalized candidate skills, experience, and projects."""
    session.query(models.CandidateSkill).filter(models.CandidateSkill.candidate_id == candidate_id).delete()
    session.query(models.CandidateExperience).filter(
        models.CandidateExperience.candidate_id == candidate_id
    ).delete()
    session.query(models.CandidateProject).filter(models.CandidateProject.candidate_id == candidate_id).delete()

    skill_rows: list[tuple[str, str]] = []
    for token in parsed.skills.technical:
        skill_rows.append((token, "technical"))
    for token in parsed.skills.soft:
        skill_rows.append((token, "soft"))
    for token in parsed.skills.cloud:
        skill_rows.append((token, "cloud"))
    for token in parsed.skills.programming_languages:
        skill_rows.append((token, "programming_language"))
    for token in parsed.skills.frameworks:
        skill_rows.append((token, "framework"))
    for token in parsed.skills.databases:
        skill_rows.append((token, "database"))
    for token in parsed.skills.ai_ml:
        skill_rows.append((token, "ai_ml"))

    seen: set[tuple[str, str]] = set()
    for name, skill_type in skill_rows:
        key = (name.lower(), skill_type)
        if key in seen:
            continue
        seen.add(key)
        session.add(
            models.CandidateSkill(
                candidate_id=candidate_id,
                skill_name=name.lower(),
                skill_type=skill_type,
            )
        )

    for item in parsed.experience:
        session.add(
            models.CandidateExperience(
                candidate_id=candidate_id,
                company=item.company,
                role=item.role,
                duration=item.duration,
                responsibilities_json=item.responsibilities,
            )
        )

    for item in parsed.projects:
        session.add(
            models.CandidateProject(
                candidate_id=candidate_id,
                title=item.title,
                technologies_json=item.technologies,
                impact=item.impact,
            )
        )
    session.flush()


def upsert_job(session: Session, jd_text: str, parsed: JobRequirementProfile) -> models.JobDescription:
    job = models.JobDescription(
        title=parsed.title,
        jd_text=jd_text,
        parsed_json=parsed.model_dump(mode="json"),
    )
    session.add(job)
    session.flush()
    return job


def save_score(session: Session, breakdown: ScoreBreakdown) -> models.CandidateJobScore:
    row = models.CandidateJobScore(
        candidate_id=breakdown.candidate_id or 0,
        job_id=breakdown.job_id or 0,
        total_score=breakdown.total_score,
        breakdown_json=breakdown.model_dump(mode="json"),
    )
    session.add(row)
    session.flush()
    return row
