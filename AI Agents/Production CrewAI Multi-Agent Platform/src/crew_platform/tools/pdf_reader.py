"""PDF reader tool."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pydantic import BaseModel

from crew_platform.tools.base import BaseTool


class PDFReaderInput(BaseModel):
    path: str
    max_pages: int = 20


class PDFReaderOutput(BaseModel):
    page_count: int
    text: str


class PDFReaderTool(BaseTool[PDFReaderInput, PDFReaderOutput]):
    name = "pdf_reader"
    description = "Read text content from PDF files"
    input_model = PDFReaderInput
    output_model = PDFReaderOutput

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    async def run(self, payload: PDFReaderInput) -> PDFReaderOutput:
        path = (self.workspace_root / payload.path).resolve()
        if self.workspace_root not in path.parents and path != self.workspace_root:
            raise ValueError("Path escapes workspace")

        reader = PdfReader(str(path))
        chunks: list[str] = []
        for page in reader.pages[: payload.max_pages]:
            chunks.append((page.extract_text() or "").strip())
        return PDFReaderOutput(page_count=len(reader.pages), text="\n".join(chunks)[:25000])
