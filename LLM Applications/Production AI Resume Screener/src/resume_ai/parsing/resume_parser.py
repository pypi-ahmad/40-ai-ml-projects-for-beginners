"""Hybrid resume parser: rules first, LLM fixup second."""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import ValidationError

from resume_ai.config.loader import AppConfig
from resume_ai.models import (
    CandidateContact,
    EducationItem,
    ExperienceItem,
    OCRMode,
    ProjectItem,
    ResumeParseResult,
    SkillSet,
)
from resume_ai.parsing.redaction import redact_text
from resume_ai.reasoning.ollama_client import OllamaLLM

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3})?[\s\-().]*\d{3}[\s\-().]*\d{3}[\s\-().]*\d{4}")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/\S+", re.IGNORECASE)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/\S+", re.IGNORECASE)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")

SKILL_KEYWORDS = {
    "python": "programming_languages",
    "java": "programming_languages",
    "javascript": "programming_languages",
    "typescript": "programming_languages",
    "pytorch": "ai_ml",
    "tensorflow": "ai_ml",
    "langchain": "ai_ml",
    "langgraph": "ai_ml",
    "fastapi": "frameworks",
    "streamlit": "frameworks",
    "react": "frameworks",
    "postgresql": "databases",
    "mysql": "databases",
    "sqlite": "databases",
    "mongodb": "databases",
    "aws": "cloud",
    "gcp": "cloud",
    "azure": "cloud",
    "kubernetes": "cloud",
    "docker": "cloud",
}


class ResumeParser:
    """Parse resume text into structured schema."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = OllamaLLM(config)

    def parse(self, text: str, ocr_mode: OCRMode, blind_mode: bool = True) -> ResumeParseResult:
        rule_result = self._parse_with_rules(text, ocr_mode)
        fixed = self._llm_fixup(rule_result, text)

        if blind_mode:
            fixed.redacted_text = redact_text(text)
            fixed.candidate.name = None
            fixed.candidate.location = None
        return fixed

    def _parse_with_rules(self, text: str, ocr_mode: OCRMode) -> ResumeParseResult:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        contact = CandidateContact(
            name=lines[0] if lines else None,
            email=(EMAIL_RE.search(text).group(0) if EMAIL_RE.search(text) else None),
            phone=(PHONE_RE.search(text).group(0) if PHONE_RE.search(text) else None),
            linkedin=(LINKEDIN_RE.search(text).group(0) if LINKEDIN_RE.search(text) else None),
            github=(GITHUB_RE.search(text).group(0) if GITHUB_RE.search(text) else None),
        )

        educations: list[EducationItem] = []
        experiences: list[ExperienceItem] = []
        projects: list[ProjectItem] = []
        certifications: list[str] = []
        awards: list[str] = []
        publications: list[str] = []

        sections = self._sectionize(lines)
        for line in sections.get("education", []):
            year = YEAR_RE.search(line)
            educations.append(
                EducationItem(
                    degree=line,
                    graduation_year=int(year.group(0)) if year else None,
                )
            )

        for line in sections.get("experience", []):
            experiences.append(
                ExperienceItem(
                    company=line,
                    role=line,
                    responsibilities=[line],
                )
            )

        for line in sections.get("projects", []):
            projects.append(ProjectItem(title=line, technologies=self._extract_skill_tokens(line)))

        for line in sections.get("certifications", []):
            certifications.append(line)

        for line in sections.get("awards", []):
            awards.append(line)

        for line in sections.get("publications", []):
            publications.append(line)

        skills = self._extract_skills(text)
        summary = " ".join(lines[:6])[:500]

        return ResumeParseResult(
            candidate=contact,
            education=educations,
            experience=experiences,
            projects=projects,
            skills=skills,
            certifications=certifications,
            awards=awards,
            publications=publications,
            summary=summary,
            ocr_mode=ocr_mode,
        )

    @staticmethod
    def _sectionize(lines: list[str]) -> dict[str, list[str]]:
        section = "experience"
        buckets: dict[str, list[str]] = defaultdict(list)
        for line in lines:
            lower = line.lower()
            if "education" in lower:
                section = "education"
                continue
            if "experience" in lower:
                section = "experience"
                continue
            if "project" in lower:
                section = "projects"
                continue
            if "certification" in lower:
                section = "certifications"
                continue
            if "award" in lower:
                section = "awards"
                continue
            if "publication" in lower:
                section = "publications"
                continue
            if "skill" in lower:
                section = "skills"
                continue
            buckets[section].append(line)
        return buckets

    @staticmethod
    def _extract_skill_tokens(text: str) -> list[str]:
        lower = text.lower()
        return [token for token in SKILL_KEYWORDS if token in lower]

    def _extract_skills(self, text: str) -> SkillSet:
        buckets: dict[str, list[str]] = defaultdict(list)
        lower = text.lower()
        for token, group in SKILL_KEYWORDS.items():
            if token in lower:
                buckets[group].append(token)
                buckets["technical"].append(token)

        return SkillSet(
            technical=sorted(set(buckets["technical"])),
            soft=["communication"] if "communication" in lower else [],
            cloud=sorted(set(buckets["cloud"])),
            programming_languages=sorted(set(buckets["programming_languages"])),
            frameworks=sorted(set(buckets["frameworks"])),
            databases=sorted(set(buckets["databases"])),
            ai_ml=sorted(set(buckets["ai_ml"])),
        )

    def _llm_fixup(self, parsed: ResumeParseResult, raw_text: str) -> ResumeParseResult:
        prompt = (
            "Fix extracted resume JSON with strict schema. Return only JSON object.\n"
            f"Schema keys: {ResumeParseResult.model_json_schema()}\n"
            f"Current parse: {parsed.model_dump_json()}\n"
            f"Raw resume text: {raw_text[:8000]}"
        )
        result = self.llm.generate_json(prompt=prompt, model=self.config.models.parser_model)
        if not result:
            return parsed

        try:
            return ResumeParseResult.model_validate(result)
        except ValidationError:
            return parsed
