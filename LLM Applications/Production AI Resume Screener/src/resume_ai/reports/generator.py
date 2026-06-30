"""Report export for markdown, HTML, JSON, and PDF."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from resume_ai.models import HiringRecommendation, ScoreBreakdown


def _render_markdown(recommendation: HiringRecommendation, score: ScoreBreakdown) -> str:
    strengths = "\n".join(f"- {item}" for item in recommendation.strengths)
    weaknesses = "\n".join(f"- {item}" for item in recommendation.weaknesses)
    evidence = "\n".join(f"- {ev.source}: {ev.snippet}" for ev in score.evidence)
    return f"""
# Candidate Hiring Report

## Recommendation
- Decision: **{recommendation.recommendation}**
- Confidence: **{recommendation.confidence_score:.2f}%**
- Total Score: **{score.total_score:.2f}**

## Strengths
{strengths}

## Weaknesses
{weaknesses}

## Matched Skills
- {", ".join(score.matched_skills)}

## Missing Skills
- {", ".join(score.missing_skills)}

## Evidence
{evidence}
""".strip()


def export_markdown(
    recommendation: HiringRecommendation,
    score: ScoreBreakdown,
    output_path: Path,
) -> Path:
    output_path.write_text(_render_markdown(recommendation, score), encoding="utf-8")
    return output_path


def export_json(
    recommendation: HiringRecommendation,
    score: ScoreBreakdown,
    output_path: Path,
) -> Path:
    payload = {
        "recommendation": recommendation.model_dump(mode="json"),
        "score": score.model_dump(mode="json"),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def export_html(
    recommendation: HiringRecommendation,
    score: ScoreBreakdown,
    output_path: Path,
) -> Path:
    template = Template(
        """
        <html>
          <head><title>Candidate Hiring Report</title></head>
          <body>
            <h1>Candidate Hiring Report</h1>
            <p><strong>Decision:</strong> {{ recommendation.recommendation }}</p>
            <p><strong>Confidence:</strong> {{ recommendation.confidence_score }}%</p>
            <p><strong>Total Score:</strong> {{ score.total_score }}</p>
            <h2>Strengths</h2>
            <ul>{% for item in recommendation.strengths %}<li>{{ item }}</li>{% endfor %}</ul>
            <h2>Weaknesses</h2>
            <ul>{% for item in recommendation.weaknesses %}<li>{{ item }}</li>{% endfor %}</ul>
          </body>
        </html>
        """
    )
    output_path.write_text(
        template.render(recommendation=recommendation.model_dump(), score=score.model_dump()),
        encoding="utf-8",
    )
    return output_path


def export_pdf(
    recommendation: HiringRecommendation,
    score: ScoreBreakdown,
    output_path: Path,
) -> Path:
    c = canvas.Canvas(str(output_path), pagesize=A4)
    y = 800
    lines = [
        "Candidate Hiring Report",
        f"Decision: {recommendation.recommendation}",
        f"Confidence: {recommendation.confidence_score:.2f}%",
        f"Total Score: {score.total_score:.2f}",
        "Strengths:",
        *[f" - {item}" for item in recommendation.strengths],
        "Weaknesses:",
        *[f" - {item}" for item in recommendation.weaknesses],
    ]
    for line in lines:
        c.drawString(40, y, line)
        y -= 16
        if y < 60:
            c.showPage()
            y = 800
    c.save()
    return output_path
