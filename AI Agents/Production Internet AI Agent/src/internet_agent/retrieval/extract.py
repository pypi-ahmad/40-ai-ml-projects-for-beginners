"""Extraction and normalization for web and PDF content."""

from __future__ import annotations

from io import BytesIO

import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader


def clean_html(html: str) -> str:
    """Extract clean text from HTML with trafilatura fallback to BeautifulSoup."""

    extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
    if extracted:
        return extracted

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_markdown(html: str) -> str:
    """Extract markdown-like output from HTML."""

    markdown = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
    )
    return markdown or clean_html(html)


def read_pdf_bytes(content: bytes) -> str:
    """Extract text from PDF bytes."""

    reader = PdfReader(BytesIO(content))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n\n".join(texts).strip()
