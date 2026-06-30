"""Export analysis outputs to markdown/html/json/pdf."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def export(payload: dict, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    html_path = out_dir / "report.html"
    pdf_path = out_dir / "report.pdf"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_content = (
        "# Multimodal Analysis Report\n\n```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n"
    )
    md_path.write_text(md_content, encoding="utf-8")

    html_content = (
        "<html><body><h1>Multimodal Analysis Report</h1>"
        f"<pre>{json.dumps(payload, indent=2)}</pre></body></html>"
    )
    html_path.write_text(html_content, encoding="utf-8")

    pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
    text = pdf.beginText(40, 740)
    text.textLine("Multimodal Analysis Report")
    for line in json.dumps(payload, indent=2).splitlines()[:60]:
        text.textLine(line)
    pdf.drawText(text)
    pdf.save()

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "pdf": str(pdf_path),
    }


if __name__ == "__main__":
    sample = {"status": "ok", "summary": "example report"}
    outputs = export(sample, Path("outputs/reports/export"))
    print(outputs)
