"""Report generation in Markdown/JSON/HTML/PDF formats."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpdf import FPDF
import markdown

from crew_platform.config import Settings
from crew_platform.orchestration.models import ReportArtifact, TaskExecution, VerificationResult


class ReportGenerator:
    """Creates report artifacts and persists required formats."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_dir = Path(settings.reports.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        run_id: str,
        objective: str,
        tasks: list[TaskExecution],
        verification: VerificationResult,
    ) -> ReportArtifact:
        sections = self._sections(tasks)
        references = self._references(tasks)
        summary = self._summary(objective, tasks, verification)

        artifact = ReportArtifact(
            run_id=run_id,
            title=f"Crew Platform Report - {run_id}",
            summary=summary,
            sections=sections,
            confidence=verification.confidence,
            references=references,
            metadata={
                "objective": objective,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "task_count": len(tasks),
                "issues": verification.issues,
            },
        )

        self._write_required_formats(artifact)
        return artifact

    def generate_on_demand(self, artifact: ReportArtifact, fmt: str) -> Path:
        fmt_lower = fmt.lower()
        if fmt_lower == "html":
            return self._write_html(artifact)
        if fmt_lower == "pdf":
            return self._write_pdf(artifact)
        if fmt_lower == "markdown":
            return self._write_markdown(artifact)
        if fmt_lower == "json":
            return self._write_json(artifact)
        raise ValueError(f"Unsupported report format: {fmt}")

    def _write_required_formats(self, artifact: ReportArtifact) -> None:
        for fmt in self.settings.reports.always_formats:
            self.generate_on_demand(artifact, fmt)

    def _write_markdown(self, artifact: ReportArtifact) -> Path:
        path = self.output_dir / f"{artifact.run_id}.md"
        lines = [
            f"# {artifact.title}",
            "",
            f"- Run ID: `{artifact.run_id}`",
            f"- Generated: `{artifact.generated_at.isoformat()}`",
            f"- Confidence: `{artifact.confidence:.3f}`",
            "",
            "## Summary",
            artifact.summary,
            "",
            "## Sections",
        ]

        for section in artifact.sections:
            lines.append(f"### {section.get('title', 'Section')}")
            lines.append(str(section.get("content", "")))
            lines.append("")

        lines.append("## References")
        for ref in artifact.references:
            lines.append(f"- {ref}")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_json(self, artifact: ReportArtifact) -> Path:
        path = self.output_dir / f"{artifact.run_id}.json"
        path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2), encoding="utf-8")
        return path

    def _write_html(self, artifact: ReportArtifact) -> Path:
        md_path = self._write_markdown(artifact)
        html = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["extra", "tables"])
        path = self.output_dir / f"{artifact.run_id}.html"
        path.write_text(
            "<html><head><meta charset='utf-8'><title>{}</title></head><body>{}</body></html>".format(
                artifact.title,
                html,
            ),
            encoding="utf-8",
        )
        return path

    def _write_pdf(self, artifact: ReportArtifact) -> Path:
        path = self.output_dir / f"{artifact.run_id}.pdf"
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, artifact.title)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, f"Run ID: {artifact.run_id}\nConfidence: {artifact.confidence:.3f}")
        pdf.multi_cell(0, 6, f"Summary:\n{artifact.summary}")
        for section in artifact.sections:
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, section.get("title", "Section"))
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, str(section.get("content", ""))[:3500])
        pdf.output(str(path))
        return path

    @staticmethod
    def _summary(objective: str, tasks: list[TaskExecution], verification: VerificationResult) -> str:
        completed = sum(task.status.value == "completed" for task in tasks)
        return (
            f"Objective: {objective}. "
            f"Completed tasks: {completed}/{len(tasks)}. "
            f"Confidence: {verification.confidence:.3f}."
        )

    @staticmethod
    def _sections(tasks: list[TaskExecution]) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        for task in tasks:
            content = ""
            if task.result:
                content = str(task.result.get("content") or task.result.get("crewai_output") or "")
            sections.append(
                {
                    "title": f"{task.task_id} - {task.agent_role}",
                    "content": content[:5000],
                    "status": task.status.value,
                    "confidence": task.confidence,
                }
            )
        return sections

    @staticmethod
    def _references(tasks: list[TaskExecution]) -> list[str]:
        refs: list[str] = []
        for task in tasks:
            if not task.result:
                continue
            content = str(task.result.get("content", ""))
            for token in content.split():
                if token.startswith("http"):
                    refs.append(token.rstrip(".,)"))
        unique = sorted(set(refs))
        return unique[:100]
