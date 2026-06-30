"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".pptx"}


def ensure_dirs(*paths: Path) -> None:
    """Create directories when absent."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def infer_media_type(path: Path) -> str:
    """Infer media type from extension."""

    ext = path.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if ext in SUPPORTED_DOC_EXTENSIONS:
        return "document"
    return "unknown"
