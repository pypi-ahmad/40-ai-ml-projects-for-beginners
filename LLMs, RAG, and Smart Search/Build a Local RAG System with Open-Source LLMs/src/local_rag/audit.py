"""Audit helpers for citation integrity and local-only dependency checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CitationAuditResult:
    """Citation integrity summary for one response."""

    valid: bool
    total_citations: int
    invalid_citations: int
    invalid_entries: list[dict[str, str]]


FORBIDDEN_IMPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai": re.compile(r"^\s*(from|import)\s+openai\b"),
    "anthropic": re.compile(r"^\s*(from|import)\s+anthropic\b"),
    "google.generativeai": re.compile(r"^\s*(from|import)\s+google\.generativeai\b"),
    "gemini_sdk": re.compile(r"^\s*(from|import)\s+google\.ai\.generativelanguage\b"),
    "azure.ai": re.compile(r"^\s*(from|import)\s+azure\.ai\b"),
    "bedrock_sdk": re.compile(r"^\s*(from|import)\s+boto3\b"),
}


def validate_citations(
    citations: list[dict[str, str]],
    retrieved: list[object],
) -> CitationAuditResult:
    """Validate that all citations map to retrieved chunk/source pairs."""

    retrieved_pairs = {
        (str(hit.metadata.get("source_path", "")), hit.chunk_id)  # type: ignore[attr-defined]
        for hit in retrieved
    }
    invalid_entries: list[dict[str, str]] = []
    for citation in citations:
        pair = (str(citation.get("source_path", "")), str(citation.get("chunk_id", "")))
        if pair not in retrieved_pairs:
            invalid_entries.append(
                {"source_path": pair[0], "chunk_id": pair[1]}
            )

    return CitationAuditResult(
        valid=not invalid_entries,
        total_citations=len(citations),
        invalid_citations=len(invalid_entries),
        invalid_entries=invalid_entries,
    )


def scan_forbidden_patterns(paths: list[Path]) -> dict[str, list[str]]:
    """Scan files for forbidden cloud-LLM dependency patterns."""

    matches: dict[str, list[str]] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:  # noqa: BLE001
            continue
        for line in lines:
            if line.strip().startswith("#"):
                continue
            for key, pattern in FORBIDDEN_IMPORT_PATTERNS.items():
                if pattern.search(line):
                    matches.setdefault(key, []).append(str(path))
    return matches
