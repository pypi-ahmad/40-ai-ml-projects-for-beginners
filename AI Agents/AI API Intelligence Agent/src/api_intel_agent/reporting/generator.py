"""Multi-format reporting utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from api_intel_agent.config import load_settings
from api_intel_agent.core.schemas import AnalyzeResponse, OutputFormat


class ReportGenerator:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.output_dir = Path(self.settings.reports.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_sections(self, response: AnalyzeResponse) -> dict[str, Any]:
        return {
            "executive_summary": response.summary,
            "statistics": response.telemetry,
            "recommendations": response.recommendations,
            "insights": response.insights,
            "sources": [item.model_dump() for item in response.sources],
            "charts": [item.model_dump() for item in response.charts],
        }

    def generate(self, response: AnalyzeResponse, output_format: OutputFormat) -> str:
        path = self.output_dir / f"report_{response.run_id}.{output_format.value if output_format != OutputFormat.MARKDOWN else 'md'}"
        data = self._base_sections(response)

        if output_format == OutputFormat.MARKDOWN:
            content = self._to_markdown(data)
            path.write_text(content)
            return str(path)
        if output_format == OutputFormat.HTML:
            content = self._to_html(data)
            path.write_text(content)
            return str(path)
        if output_format == OutputFormat.JSON:
            path.write_text(json.dumps(data, indent=2, default=str))
            return str(path)
        if output_format == OutputFormat.CSV:
            self._to_csv(path, data)
            return str(path)
        if output_format == OutputFormat.PDF:
            self._to_pdf(path, data)
            return str(path)

        content = self._to_markdown(data)
        path.write_text(content)
        return str(path)

    def _to_markdown(self, data: dict[str, Any]) -> str:
        lines = ["# Executive Summary", data["executive_summary"], "", "## Statistics"]
        for key, value in data["statistics"].items():
            lines.append(f"- {key}: {value}")
        lines.append("\n## Recommendations")
        lines.extend(f"- {item}" for item in data["recommendations"])
        lines.append("\n## Insights")
        lines.extend(f"- {item}" for item in data["insights"])
        lines.append("\n## Sources")
        lines.extend(f"- {src['provider']} :: {src['endpoint']}" for src in data["sources"])
        return "\n".join(lines)

    def _to_html(self, data: dict[str, Any]) -> str:
        markdown = self._to_markdown(data)
        paragraphs = "".join(f"<p>{line}</p>" for line in markdown.splitlines() if line)
        return f"<html><body>{paragraphs}</body></html>"

    def _to_csv(self, path: Path, data: dict[str, Any]) -> None:
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["section", "value"])
            writer.writerow(["executive_summary", data["executive_summary"]])
            for key, value in data["statistics"].items():
                writer.writerow([f"statistics.{key}", value])
            for item in data["recommendations"]:
                writer.writerow(["recommendation", item])
            for item in data["insights"]:
                writer.writerow(["insight", item])

    def _to_pdf(self, path: Path, data: dict[str, Any]) -> None:
        pdf = canvas.Canvas(str(path), pagesize=letter)
        y = 760
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(72, y, "Executive Summary")
        y -= 20
        pdf.setFont("Helvetica", 10)
        for line in self._to_markdown(data).splitlines():
            if y < 72:
                pdf.showPage()
                y = 760
            pdf.drawString(72, y, line[:110])
            y -= 14
        pdf.save()
