"""Report export engine for Markdown/HTML/PDF/JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from markdown import markdown


class ReportExporter:
    """Export workflow reports in multiple formats."""

    def __init__(self, output_dir: str = "artifacts/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_markdown(self, workflow_id: str, markdown_text: str) -> Path:
        path = self.output_dir / f"{workflow_id}.md"
        path.write_text(markdown_text, encoding="utf-8")
        return path

    def export_html(self, workflow_id: str, markdown_text: str) -> Path:
        html = markdown(markdown_text)
        path = self.output_dir / f"{workflow_id}.html"
        path.write_text(html, encoding="utf-8")
        return path

    def export_pdf(self, workflow_id: str, markdown_text: str) -> Path:
        html = markdown(markdown_text)
        path = self.output_dir / f"{workflow_id}.pdf"
        try:
            from weasyprint import HTML

            HTML(string=html).write_pdf(str(path))
        except Exception:
            # Graceful fallback when Cairo/Pango system deps are missing.
            path.write_text("PDF export unavailable in this environment.", encoding="utf-8")
        return path

    def export_json(self, workflow_id: str, payload: dict[str, Any]) -> Path:
        path = self.output_dir / f"{workflow_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def export_all(
        self, workflow_id: str, markdown_text: str, payload: dict[str, Any]
    ) -> dict[str, str]:
        """Export report in all supported formats."""

        return {
            "markdown": str(self.export_markdown(workflow_id, markdown_text)),
            "html": str(self.export_html(workflow_id, markdown_text)),
            "pdf": str(self.export_pdf(workflow_id, markdown_text)),
            "json": str(self.export_json(workflow_id, payload)),
        }
