"""Blind hiring redaction helpers."""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3})?[\s\-().]*\d{3}[\s\-().]*\d{3}[\s\-().]*\d{4}")
LINK_RE = re.compile(r"https?://\S+|www\.\S+")


def redact_text(text: str) -> str:
    """Remove common PII tokens for blind screening."""
    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = LINK_RE.sub("[REDACTED_LINK]", redacted)
    return redacted
