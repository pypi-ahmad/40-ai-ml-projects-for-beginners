"""Job description parser and terminology normalization."""

from __future__ import annotations

import re

from resume_ai.config.loader import AppConfig
from resume_ai.models import JobRequirementProfile
from resume_ai.reasoning.ollama_client import OllamaLLM

ALIAS_MAP = {
    "reactjs": "react",
    "react.js": "react",
    "react framework": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "llm": "large language models",
}

SKILL_HINTS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "fastapi",
    "streamlit",
    "langchain",
    "langgraph",
    "k8s",
    "kubernetes",
    "docker",
    "aws",
    "gcp",
    "azure",
    "pytorch",
    "tensorflow",
    "mlflow",
]


class JobDescriptionParser:
    """Extract and normalize job requirements."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = OllamaLLM(config)

    def parse(self, jd_text: str) -> JobRequirementProfile:
        rule = self._parse_rules(jd_text)
        fix = self._llm_fixup(rule, jd_text)
        return self._normalize(fix)

    def _parse_rules(self, jd_text: str) -> JobRequirementProfile:
        lower = jd_text.lower()
        required = [skill for skill in SKILL_HINTS if skill in lower]
        preferred = [skill for skill in ["graphql", "snowflake", "airflow"] if skill in lower]
        keywords = sorted(set(re.findall(r"[a-zA-Z][a-zA-Z+.#-]{2,}", lower)))
        responsibilities = [line.strip("- ") for line in jd_text.splitlines() if line.strip().startswith("-")]

        return JobRequirementProfile(
            title=self._extract_title(jd_text),
            required_skills=required,
            preferred_skills=preferred,
            experience_requirements=self._extract_experience(jd_text),
            education_requirements=self._extract_education(jd_text),
            responsibilities=responsibilities,
            keywords=keywords[:100],
            technologies=required + preferred,
            soft_skills=[
                token
                for token in ["communication", "leadership", "ownership", "collaboration"]
                if token in lower
            ],
        )

    def _llm_fixup(self, parsed: JobRequirementProfile, jd_text: str) -> JobRequirementProfile:
        prompt = (
            "Refine job description extraction into strict JSON. Return JSON only.\n"
            f"Schema: {JobRequirementProfile.model_json_schema()}\n"
            f"Current parse: {parsed.model_dump_json()}\n"
            f"JD text: {jd_text[:10000]}"
        )
        response = self.llm.generate_json(prompt, model=self.config.models.extraction_model)
        if not response:
            return parsed
        try:
            return JobRequirementProfile.model_validate(response)
        except Exception:
            return parsed

    def _normalize(self, profile: JobRequirementProfile) -> JobRequirementProfile:
        profile.required_skills = self._normalize_list(profile.required_skills)
        profile.preferred_skills = self._normalize_list(profile.preferred_skills)
        profile.technologies = self._normalize_list(profile.technologies)
        profile.keywords = self._normalize_list(profile.keywords)
        return profile

    @staticmethod
    def _normalize_list(values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            token = value.strip().lower()
            token = ALIAS_MAP.get(token, token)
            cleaned.append(token)
        return sorted(set(cleaned))

    @staticmethod
    def _extract_title(text: str) -> str | None:
        for line in text.splitlines()[:5]:
            if "engineer" in line.lower() or "developer" in line.lower() or "scientist" in line.lower():
                return line.strip()
        return None

    @staticmethod
    def _extract_experience(text: str) -> list[str]:
        return re.findall(r"\b\d+\+?\s+years?\b", text.lower())

    @staticmethod
    def _extract_education(text: str) -> list[str]:
        buckets: list[str] = []
        lower = text.lower()
        if "bachelor" in lower:
            buckets.append("bachelor")
        if "master" in lower:
            buckets.append("master")
        if "phd" in lower or "doctorate" in lower:
            buckets.append("phd")
        return buckets
