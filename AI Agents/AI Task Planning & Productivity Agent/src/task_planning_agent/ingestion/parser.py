"""Messy input parser and task candidate normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import dateparser


@dataclass(slots=True)
class ParsedTaskCandidate:
    raw_line: str
    title: str
    description: str
    deadline_text: str | None
    estimated_minutes: int | None
    people: list[str]
    project: str | None


TASK_LINE_PATTERN = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(?P<body>.+)$")
ESTIMATE_PATTERN = re.compile(r"(?P<val>\d+)\s*(?:min|mins|minutes|h|hr|hours)", re.IGNORECASE)
DEADLINE_HINT_PATTERN = re.compile(
    r"(?:by|before|due|deadline|tomorrow|today|next\s+\w+|\d{1,2}/\d{1,2}(?:/\d{2,4})?)",
    re.IGNORECASE,
)
PERSON_PATTERN = re.compile(r"@([\w.-]+)")
PROJECT_PATTERN = re.compile(r"#([\w-]+)")


def _extract_estimate_minutes(text: str) -> int | None:
    match = ESTIMATE_PATTERN.search(text)
    if not match:
        return None
    val = int(match.group("val"))
    token = match.group(0).lower()
    if "h" in token and "min" not in token:
        return val * 60
    return val


def _extract_deadline_text(text: str) -> str | None:
    if not DEADLINE_HINT_PATTERN.search(text):
        return None
    parsed = dateparser.parse(text)
    if parsed is None:
        return None
    return parsed.isoformat()


def parse_messy_input(raw_input: str) -> list[ParsedTaskCandidate]:
    """Convert messy user text into task candidates."""

    candidates: list[ParsedTaskCandidate] = []
    lines = [line.strip() for line in raw_input.splitlines() if line.strip()]

    for line in lines:
        match = TASK_LINE_PATTERN.match(line)
        body = match.group("body") if match else line
        title = body.split(".")[0][:120].strip()
        deadline_text = _extract_deadline_text(body)
        est_minutes = _extract_estimate_minutes(body)
        people = PERSON_PATTERN.findall(body)
        project_match = PROJECT_PATTERN.search(body)
        project = project_match.group(1) if project_match else None

        candidates.append(
            ParsedTaskCandidate(
                raw_line=line,
                title=title,
                description=body,
                deadline_text=deadline_text,
                estimated_minutes=est_minutes,
                people=people,
                project=project,
            )
        )

    return candidates


def normalize_notes_blocks(text_blocks: Iterable[str]) -> str:
    """Normalize notes/transcripts/slack/email text blocks into plain text."""

    merged = "\n".join(block.strip() for block in text_blocks if block.strip())
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()
