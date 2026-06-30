"""Report generation service for Markdown/HTML/PDF/JSON exports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF

from internet_agent.config import Settings


class ReportService:
    """Generate export artifacts from run payloads."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_dir = Path(settings.reports.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_name(self, session_id: str) -> str:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"report_{session_id}_{stamp}"

    def generate_markdown(self, session_id: str, payload: dict[str, Any]) -> Path:
        name = self._base_name(session_id)
        path = self.output_dir / f"{name}.md"

        citations = "\n".join(
            f"- [{item.get('title', 'source')}]({item.get('url', '')})"
            for item in payload.get("citations", [])
        )
        trace = "\n".join(f"- {row}" for row in payload.get("reasoning_trace", []))

        content = (
            f"# Internet Agent Report\n\n"
            f"- Session: `{session_id}`\n"
            f"- Query: {payload.get('query', '')}\n"
            f"- Timestamp: {payload.get('timestamp', '')}\n"
            f"- Confidence: {payload.get('confidence', 0.0):.3f}\n\n"
            f"## Answer\n\n{payload.get('answer', '')}\n\n"
            f"## Citations\n\n{citations or '- none'}\n\n"
            f"## Verification\n\n"
            f"- Hallucination risk: {payload.get('hallucination_risk', 'unknown')}\n"
            f"- Missing info: {payload.get('missing_info', [])}\n"
            f"- Conflicts: {payload.get('conflicts', [])}\n\n"
            f"## Reasoning Trace (Tool/Action)\n\n{trace or '- none'}\n"
        )
        path.write_text(content, encoding="utf-8")
        return path

    def generate_html(self, session_id: str, payload: dict[str, Any]) -> Path:
        md_path = self.generate_markdown(session_id, payload)
        html_path = md_path.with_suffix(".html")
        markdown = md_path.read_text(encoding="utf-8")
        html = (
            "<html><body><pre style='white-space:pre-wrap;font-family:monospace;'>"
            + markdown
            + "</pre></body></html>"
        )
        html_path.write_text(html, encoding="utf-8")
        return html_path

    def generate_json(self, session_id: str, payload: dict[str, Any]) -> Path:
        path = self.output_dir / f"{self._base_name(session_id)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return path

    def generate_pdf(self, session_id: str, payload: dict[str, Any]) -> Path:
        path = self.output_dir / f"{self._base_name(session_id)}.pdf"

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        lines = [
            "Internet Agent Report",
            f"Session: {session_id}",
            f"Query: {payload.get('query', '')}",
            f"Confidence: {payload.get('confidence', 0.0):.3f}",
            "",
            "Answer:",
            payload.get("answer", ""),
            "",
            "Citations:",
        ]
        for citation in payload.get("citations", []):
            lines.append(f"- {citation.get('title', 'source')} -> {citation.get('url', '')}")

        for line in lines:
            safe_line = line.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 7, safe_line)

        pdf.output(str(path))
        return path

    def generate(self, session_id: str, payload: dict[str, Any], fmt: str) -> Path:
        fmt = fmt.lower()
        if fmt == "markdown":
            return self.generate_markdown(session_id, payload)
        if fmt == "html":
            return self.generate_html(session_id, payload)
        if fmt == "json":
            return self.generate_json(session_id, payload)
        if fmt == "pdf":
            return self.generate_pdf(session_id, payload)
        raise ValueError(f"Unsupported report format: {fmt}")
