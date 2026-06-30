"""Run end-to-end demo on sample CSV without Google API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ai_spreadsheet_analytics.analytics import AnalyticsEngine
from ai_spreadsheet_analytics.cleaning import DataCleaner
from ai_spreadsheet_analytics.config import Settings
from ai_spreadsheet_analytics.insights import InsightEngine
from ai_spreadsheet_analytics.llm.ollama_rest import OllamaRESTClient
from ai_spreadsheet_analytics.quality import DataQualityProfiler
from ai_spreadsheet_analytics.reporting import ReportGenerator
from ai_spreadsheet_analytics.schemas import CleaningStrategy


def main() -> None:
    load_dotenv()
    settings = Settings()
    settings.ensure_directories()

    df = pd.read_csv(Path("data/samples/retail_sales.csv"))

    quality = DataQualityProfiler().profile("retail_sales", df)
    cleaned = DataCleaner().clean("retail_sales", df, CleaningStrategy())
    eda = AnalyticsEngine().run_full_eda(cleaned.cleaned)

    insight = InsightEngine(OllamaRESTClient(settings.ollama_base_url), default_temperature=0.0).generate(
        analytics_payload=eda,
        role="executive",
        model=settings.ollama_primary_model,
    )

    artifacts = ReportGenerator(settings.report_dir).generate(
        title="Retail Offline Demo",
        insights_markdown=insight.summary,
        tables={
            "cleaned_preview": cleaned.cleaned.head(50),
            "numeric_summary": pd.DataFrame(eda["summary"]["numeric_summary"]),
        },
    )

    print("Quality metrics:", quality.metrics)
    print("Report artifacts:", artifacts.model_dump())


if __name__ == "__main__":
    main()
