"""Compare qwen3.5:4b and granite4.1:3b for latency and judged quality."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from ai_spreadsheet_analytics.config import Settings
from ai_spreadsheet_analytics.insights import InsightEngine
from ai_spreadsheet_analytics.judge import LLMJudge
from ai_spreadsheet_analytics.llm.ollama_rest import OllamaRESTClient
from ai_spreadsheet_analytics.state_store import SQLiteStateStore


def main() -> None:
    load_dotenv()
    settings = Settings()
    settings.ensure_directories()

    state = SQLiteStateStore(settings.state_db_path)
    client = OllamaRESTClient(settings.ollama_base_url)
    insight = InsightEngine(client, default_temperature=0.0)
    judge = LLMJudge(client, judge_model=settings.ollama_judge_model, state_store=state)

    payload = {
        "kpis": {"total_revenue": 1250000, "revenue_growth_rate": 0.084, "unique_customers": 1245},
        "trend_detection": {"trend": "upward", "slope": 4.2},
        "anomalies": {"count": 2, "samples": [{"month": "2024-10"}, {"month": "2025-01"}]},
    }

    rows: list[dict] = []
    for model in [settings.ollama_primary_model, settings.ollama_secondary_model]:
        try:
            packet = insight.generate(payload, role="executive", model=model)
            eval_report = judge.evaluate(
                model_evaluated=model,
                question="Summarize business performance and risks.",
                deterministic_evidence=payload,
                model_answer=packet.summary,
            )
            rows.append(
                {
                    "model": model,
                    "latency_ms": packet.latency_ms,
                    "token_estimate": packet.token_estimate,
                    "judge_scores": eval_report.scores.model_dump(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "model": model,
                    "error": str(exc),
                    "latency_ms": None,
                    "token_estimate": None,
                    "judge_scores": None,
                }
            )

    output = Path("data/artifacts/model_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
