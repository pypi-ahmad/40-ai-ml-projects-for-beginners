"""End-to-end AI SQL assistant orchestration pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ai_sql_assistant.config import AppSettings, get_settings
from ai_sql_assistant.execution.executor import QueryExecutor
from ai_sql_assistant.explanation.explainer import SQLExplainer
from ai_sql_assistant.generation.direct import DirectSQLGenerator
from ai_sql_assistant.generation.langchain_generator import LangChainSQLGenerator
from ai_sql_assistant.logging_utils import logger
from ai_sql_assistant.memory.followup import resolve_followup_sql
from ai_sql_assistant.memory.store import AppStateStore
from ai_sql_assistant.schema.context import schema_context_text
from ai_sql_assistant.schema.introspector import inspect_database
from ai_sql_assistant.schema.summarizer import (
    load_cached_summary,
    save_schema_summary,
    schema_signature,
    summarize_schema,
)
from ai_sql_assistant.types import (
    AssistantResponse,
    ExecutionResult,
    GenerationCandidate,
    QueryRequest,
    ValidationIssue,
    ValidationReport,
)
from ai_sql_assistant.utils.sql_utils import fix_common_sqlite_patterns
from ai_sql_assistant.validation.validator import SQLValidator
from ai_sql_assistant.visualization.recommender import recommend_visualizations


class AISQLAssistant:
    """Production-style assistant that maps NL questions to validated SQL analytics."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

        active_db = self.settings.database.active_db_path
        self.active_db_path: Path = active_db
        self.db_uri = f"sqlite:///{active_db.as_posix()}"

        self.schema_report = inspect_database(active_db)
        self.schema_context = schema_context_text(self.schema_report)

        signature = schema_signature(self.schema_report)
        cache_path = Path("artifacts/reports") / f"schema_summary_{signature[:12]}.json"
        cached = load_cached_summary(cache_path)
        if cached is None:
            self.schema_summary = summarize_schema(self.schema_report)
            save_schema_summary(self.schema_summary, signature)
        else:
            self.schema_summary = cached

        self.validator = SQLValidator(
            schema_report=self.schema_report,
            allow_multi_statement=self.settings.safety.allow_multi_statement,
            strict_join_checks=self.settings.safety.strict_join_checks,
        )
        self.executor = QueryExecutor(
            db_path=active_db,
            max_rows=self.settings.safety.max_rows,
            max_query_seconds=self.settings.safety.max_query_seconds,
        )
        self.explainer = SQLExplainer(self.settings)
        self.direct_generator = DirectSQLGenerator(self.settings)
        self.langchain_generator = LangChainSQLGenerator(self.settings, self.db_uri)
        self.store = AppStateStore(self.settings.database.app_state_db_path)

    def close(self) -> None:
        """Close clients/resources."""
        self.direct_generator.close()
        self.explainer.close()

    def ask(
        self,
        request: QueryRequest,
        approach: str = "langchain",
        model: str | None = None,
        generate_explanation: bool = True,
        enable_recovery: bool = True,
    ) -> AssistantResponse:
        """Process one natural-language analytics request end-to-end."""
        model_name = model or self.settings.models.generator_model
        memory_context = self.store.conversation_context(request.conversation_id)
        previous_sql = self.store.last_sql(request.conversation_id)
        followup_sql = resolve_followup_sql(request.question, previous_sql)

        if followup_sql:
            generation = GenerationCandidate(
                sql=followup_sql,
                approach=approach,
                model=model_name,
                prompt_name=f"{request.persona}|followup_rewrite",
                latency_ms=0.0,
                fallback_used=True,
                raw_response="rule_based_followup",
            )
        else:
            generation = self._generate_candidate(
                request=request,
                approach=approach,
                model=model_name,
                memory_context=memory_context,
            )

        generation.sql = fix_common_sqlite_patterns(generation.sql)
        validation = self.validator.validate(generation.sql)
        if validation.is_valid:
            execution = self.executor.execute(validation.normalized_sql)
        else:
            execution = ExecutionResult(
                status="blocked",
                sql=generation.sql,
                error_message=self._render_validation_errors(validation),
            )

        if enable_recovery and ((not validation.is_valid) or execution.status == "error"):
            generation, validation, execution = self._recover_from_failures(
                request=request,
                memory_context=memory_context,
                initial_generation=generation,
                initial_validation=validation,
                initial_execution=execution,
                primary_approach=approach,
                primary_model=model_name,
            )

        if not validation.is_valid or execution.status == "blocked":
            execution = ExecutionResult(
                status="blocked",
                sql=generation.sql,
                error_message=self._render_validation_errors(validation),
            )
            explanation = "SQL blocked by safety/semantic validator."
            business = "No business result because query did not pass validation."
        elif execution.status == "error":
            explanation = f"SQL execution failed: {execution.error_message}"
            business = "No business result because query execution failed."
        else:
            if generate_explanation:
                explanation, business = self.explainer.explain(
                    question=request.question,
                    sql=execution.sql,
                    explain_plan=execution.explain_plan,
                    model=self.settings.models.comparison_model,
                )
            else:
                explanation = "Explanation generation disabled for this run."
                business = "Business interpretation disabled for this run."

        viz = recommend_visualizations(execution.rows)

        self.store.add_history(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            question=request.question,
            sql=execution.sql,
            approach=generation.approach,
            model=generation.model,
            status=execution.status,
            latency_ms=generation.latency_ms + execution.execution_time_ms,
            row_count=execution.row_count,
        )
        self.store.add_turn(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            question=request.question,
            sql=execution.sql,
            explanation=explanation,
        )

        return AssistantResponse(
            request=request,
            generation=generation,
            validation=validation,
            execution=execution,
            explanation=explanation,
            business_interpretation=business,
            visualization_options=viz,
        )

    def schema_browser_payload(self) -> dict[str, Any]:
        """Return schema + summary payload for app schema browser."""
        return {
            "schema_report": self.schema_report,
            "schema_summary": self.schema_summary,
        }

    def history_frame(self, limit: int = 100) -> pd.DataFrame:
        """Return recent query history."""
        return self.store.history(limit)

    def favorites_frame(self) -> pd.DataFrame:
        """Return favorites."""
        return self.store.list_favorites()

    def add_favorite(self, label: str, question: str, sql: str) -> None:
        """Persist favorite query record."""
        self.store.add_favorite(label=label, question=question, sql=sql)

    def dashboard_stats(self) -> dict[str, Any]:
        """Return aggregate history statistics."""
        return self.store.dashboard_stats()

    def export_result_csv(self, rows: list[dict[str, Any]], path: Path) -> Path:
        """Export result set to CSV."""
        frame = pd.DataFrame(rows)
        frame.to_csv(path, index=False)
        return path

    def export_result_excel(self, rows: list[dict[str, Any]], path: Path) -> Path:
        """Export result set to XLSX."""
        frame = pd.DataFrame(rows)
        frame.to_excel(path, index=False)
        return path

    def export_sql_report(self, response: AssistantResponse, path: Path) -> Path:
        """Export query report as JSON artifact."""
        payload = json.loads(response.model_dump_json())
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _render_validation_errors(validation: ValidationReport) -> str:
        issues = [f"[{issue.code}] {issue.message}" for issue in validation.issues]
        return "\n".join(issues) if issues else "Unknown validation failure."

    def _generate_candidate(
        self,
        request: QueryRequest,
        approach: str,
        model: str,
        memory_context: str,
    ) -> GenerationCandidate:
        """Generate one SQL candidate for selected approach/model."""
        if approach == "langchain":
            started = time.perf_counter()
            generation = self.langchain_generator.generate(
                question=request.question,
                persona=request.persona,
                model=model,
            )
            generation.latency_ms = (time.perf_counter() - started) * 1000.0
            return generation

        return self.direct_generator.generate(
            question=request.question,
            schema_context=self.schema_context,
            persona=request.persona,
            memory_context=memory_context,
            model=model,
        )

    def _recover_from_failures(
        self,
        request: QueryRequest,
        memory_context: str,
        initial_generation: GenerationCandidate,
        initial_validation: ValidationReport,
        initial_execution: ExecutionResult,
        primary_approach: str,
        primary_model: str,
    ) -> tuple[GenerationCandidate, ValidationReport, ExecutionResult]:
        """Try fallback generation paths when validation/execution fails."""
        candidates = [
            (primary_approach, primary_model),
            ("langchain", primary_model),
            ("direct", primary_model),
            ("langchain", self.settings.models.comparison_model),
            ("direct", self.settings.models.comparison_model),
        ]

        seen: set[tuple[str, str]] = set()
        last_generation = initial_generation
        last_validation = initial_validation
        last_execution = initial_execution

        for fallback_approach, fallback_model in candidates:
            key = (fallback_approach, fallback_model)
            if key in seen:
                continue
            seen.add(key)

            if (
                fallback_approach == initial_generation.approach
                and fallback_model == initial_generation.model
                and not initial_generation.fallback_used
            ):
                continue

            try:
                generation = self._generate_candidate(
                    request=request,
                    approach=fallback_approach,
                    model=fallback_model,
                    memory_context=memory_context,
                )
            except Exception as exc:
                logger.warning(
                    "Fallback generation failed approach={} model={}: {}",
                    fallback_approach,
                    fallback_model,
                    exc,
                )
                continue

            generation.fallback_used = True
            generation.prompt_name = f"{generation.prompt_name}|recovery"
            generation.sql = fix_common_sqlite_patterns(generation.sql)
            validation = self.validator.validate(generation.sql)

            if not validation.is_valid:
                last_generation, last_validation = generation, validation
                last_execution = ExecutionResult(
                    status="blocked",
                    sql=generation.sql,
                    error_message=self._render_validation_errors(validation),
                )
                continue

            execution = self.executor.execute(validation.normalized_sql)
            last_generation, last_validation, last_execution = generation, validation, execution
            if execution.status == "success":
                return generation, validation, execution

        return last_generation, last_validation, last_execution


def create_assistant(settings: AppSettings | None = None) -> AISQLAssistant:
    """Factory for assistant instance."""
    return AISQLAssistant(settings=settings)
