"""End-to-end analytics pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
from loguru import logger

from ai_spreadsheet_analytics.analytics import AnalyticsEngine
from ai_spreadsheet_analytics.cleaning import DataCleaner
from ai_spreadsheet_analytics.config import Settings
from ai_spreadsheet_analytics.connectors.auth import build_service_account_client
from ai_spreadsheet_analytics.connectors.google_sheets import GoogleSheetsLoader, SheetLoadRequest
from ai_spreadsheet_analytics.insights import InsightEngine
from ai_spreadsheet_analytics.llm.ollama_rest import OllamaRESTClient
from ai_spreadsheet_analytics.quality import DataQualityProfiler
from ai_spreadsheet_analytics.reporting import ReportGenerator
from ai_spreadsheet_analytics.schemas import CleaningStrategy, InsightPacket
from ai_spreadsheet_analytics.state_store import SQLiteStateStore
from ai_spreadsheet_analytics.visualization import VisualizationEngine
from ai_spreadsheet_analytics.writeback import GoogleSheetsWriteBack


class AnalyticsPipeline:
    """Orchestrates ingestion -> quality -> cleaning -> analytics -> insights -> report."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()

        self.state_store = SQLiteStateStore(settings.state_db_path)
        self.client = None
        self.loader = None
        if settings.google_service_account_json and settings.google_service_account_json.exists():
            self.client = build_service_account_client(settings.google_service_account_json, settings.scopes)
            self.loader = GoogleSheetsLoader(self.client, settings.cache_dir, self.state_store)
        else:
            logger.warning(
                "Google service account path missing or not found. Google Sheets ingestion/write-back disabled."
            )

        self.quality = DataQualityProfiler()
        self.cleaner = DataCleaner()
        self.analytics = AnalyticsEngine()
        self.viz = VisualizationEngine()

        self.llm_rest = OllamaRESTClient(settings.ollama_base_url)
        self.insights = InsightEngine(self.llm_rest, default_temperature=settings.ollama_temperature)
        self.reports = ReportGenerator(settings.report_dir)
        self.writeback = GoogleSheetsWriteBack(self.loader) if self.loader else None

    def load_bundle(self, spreadsheet_ids: list[str], worksheets: list[str]) -> pd.DataFrame:
        """Load all requested sheets and return unified dataframe."""
        requests: list[SheetLoadRequest] = []
        if self.loader is None:
            raise RuntimeError(
                "GoogleSheetsLoader unavailable. Configure GOOGLE_SERVICE_ACCOUNT_JSON for live sheet loading."
            )
        for sid in spreadsheet_ids:
            if worksheets:
                for ws in worksheets:
                    requests.append(SheetLoadRequest(spreadsheet_id=sid, worksheet_title=ws))
            else:
                for item in self.loader.inspect_spreadsheet(sid):
                    requests.append(
                        SheetLoadRequest(spreadsheet_id=sid, worksheet_title=str(item["worksheet_title"]))
                    )

        bundle = self.loader.load_batch(requests=requests, use_cache=True)
        return bundle.combined()

    def run(
        self,
        dataframe: pd.DataFrame,
        cleaning_strategy: CleaningStrategy,
        role: str,
        model: str,
        report_title: str,
    ) -> dict[str, Any]:
        """Run full analytical workflow on dataframe."""
        quality_report = self.quality.profile("combined", dataframe)
        cleaning_result = self.cleaner.clean("combined", dataframe, cleaning_strategy)
        eda = self.analytics.run_full_eda(cleaning_result.cleaned)
        try:
            insight_packet = self.insights.generate(
                analytics_payload=eda,
                role=role,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM insight generation failed: {}", exc)
            insight_packet = InsightPacket(
                prompt_role=role,
                model=model,
                summary=(
                    "LLM unavailable; returned deterministic analytics only. "
                    "Start Ollama and pull configured model to enable narrative insights."
                ),
                findings=[],
                recommendations=["Start Ollama daemon and ensure model exists locally."],
                deterministic_evidence=eda,
                latency_ms=0.0,
                token_estimate=0,
            )

        tables = {
            "cleaned_preview": cleaning_result.cleaned.head(50),
            "numeric_summary": pd.DataFrame(eda["summary"]["numeric_summary"]),
        }
        artifacts = self.reports.generate(
            title=report_title,
            insights_markdown=insight_packet.summary,
            tables=tables,
        )

        result = {
            "quality_report": {
                "metrics": quality_report.metrics,
                "issues": [asdict(issue) for issue in quality_report.issues],
            },
            "cleaning_actions": cleaning_result.actions,
            "eda": eda,
            "insight": {
                "model": insight_packet.model,
                "summary": insight_packet.summary,
                "findings": insight_packet.findings,
                "recommendations": insight_packet.recommendations,
                "latency_ms": insight_packet.latency_ms,
                "token_estimate": insight_packet.token_estimate,
            },
            "report_artifacts": {
                key: str(value)
                for key, value in artifacts.model_dump().items()
                if value is not None
            },
        }

        metrics_path = Path(self.settings.report_dir) / "latest_pipeline_result.json"
        metrics_path.write_bytes(
            orjson.dumps(
                result,
                option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
            )
        )
        return result
