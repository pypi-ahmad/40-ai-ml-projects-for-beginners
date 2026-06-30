"""Gradio callback handlers and service registry for the multi-tab application."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.benchmarking import BENCHMARK_MODELS, BenchmarkRunner, format_benchmark_table
from src.chat import ChatEngine
from src.config import get_config
from src.document_analyzer import DocumentAnalyzer
from src.ollama_client import OllamaClient
from src.sentiment import SentimentAnalyzer
from src.summarization import Summarizer
from src.translation import SUPPORTED_LANGUAGES, Translator
from src.visualization import BenchmarkVisualizer

logger = logging.getLogger(__name__)


CHAT_MODELS = ["qwen3.5:4b", "granite4.1:3b"]
OCR_MODELS = ["glm-ocr:latest", "deepseek-ocr:latest"]
SUMMARY_MODELS = ["granite4.1:3b", "qwen3.5:4b"]
SENTIMENT_MODELS = ["qwen3.5:2b", "qwen3.5:4b"]


@dataclass
class ServiceRegistry:
    """Lazy service factory to avoid loading every model at startup."""

    sentiment_services: dict[str, SentimentAnalyzer]
    summarization_services: dict[str, Summarizer]
    translation_services: dict[str, Translator]
    chat_services: dict[str, ChatEngine]
    document_services: dict[tuple[str, str], DocumentAnalyzer]

    @classmethod
    def create(cls) -> ServiceRegistry:
        """Create empty lazy registry."""

        return cls({}, {}, {}, {}, {})

    def get_sentiment(self, model: str) -> SentimentAnalyzer:
        """Return cached sentiment analyzer for model."""

        if model not in self.sentiment_services:
            self.sentiment_services[model] = SentimentAnalyzer(model=model)
        return self.sentiment_services[model]

    def get_summarizer(self, model: str) -> Summarizer:
        """Return cached summarizer for model."""

        if model not in self.summarization_services:
            self.summarization_services[model] = Summarizer(model=model)
        return self.summarization_services[model]

    def get_translator(self, model: str) -> Translator:
        """Return cached translator for model."""

        if model not in self.translation_services:
            self.translation_services[model] = Translator(model=model)
        return self.translation_services[model]

    def get_chat(self, model: str) -> ChatEngine:
        """Return cached chat engine for model."""

        if model not in self.chat_services:
            self.chat_services[model] = ChatEngine(model=model)
        return self.chat_services[model]

    def get_document_analyzer(self, ocr_model: str, qa_model: str) -> DocumentAnalyzer:
        """Return cached document analyzer for (ocr, qa) model pair."""

        key = (ocr_model, qa_model)
        if key not in self.document_services:
            self.document_services[key] = DocumentAnalyzer(ocr_model=ocr_model, qa_model=qa_model)
        return self.document_services[key]

    def close(self) -> None:
        """Close all cached clients on shutdown."""

        for service_map in [
            self.sentiment_services,
            self.summarization_services,
            self.translation_services,
            self.chat_services,
            self.document_services,
        ]:
            for service in service_map.values():
                service.close()


class AppHandlers:
    """Shared callback logic used by Gradio app and tests."""

    def __init__(self, registry: ServiceRegistry | None = None) -> None:
        self.registry = registry or ServiceRegistry.create()
        self._models_cache: set[str] = set()
        self._models_cache_updated_at = 0.0
        self._models_cache_ttl_s = 15.0

    @staticmethod
    def _error_markdown(title: str, message: str) -> str:
        """Format consistent, user-friendly error markdown."""

        return f"### {title}\n\n**Error:** {message}"

    def _refresh_model_cache(self, force: bool = False) -> set[str]:
        """Refresh available model names from local Ollama with light cache."""

        now = time.time()
        should_refresh = force or (now - self._models_cache_updated_at >= self._models_cache_ttl_s)
        if not should_refresh:
            return self._models_cache

        client = OllamaClient()
        try:
            self._models_cache = set(client.list_models())
            self._models_cache_updated_at = now
        finally:
            client.close()
        return self._models_cache

    def _ensure_model_available(self, model: str) -> str | None:
        """Return human-readable error when requested model is unavailable."""

        if model in self._refresh_model_cache():
            return None
        return (
            f"Model '{model}' is not available in local Ollama. "
            f"Pull it first: `ollama pull {model}`."
        )

    @staticmethod
    def _safe_json(payload: dict[str, Any]) -> str:
        """Serialize payload for debug widgets."""

        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _normalize_history_state(
        history_by_model: dict[str, list[dict[str, str]]] | None,
    ) -> dict[str, list[dict[str, str]]]:
        """Normalize chat state shape used by Gradio state container."""

        if not isinstance(history_by_model, dict):
            return {}
        normalized: dict[str, list[dict[str, str]]] = {}
        for model, history in history_by_model.items():
            if isinstance(model, str) and isinstance(history, list):
                normalized[model] = history
        return normalized

    def handle_sentiment(self, text: str, model: str) -> tuple[str, float, str, str]:
        """Sentiment tab callback."""

        model_error = self._ensure_model_available(model)
        if model_error:
            return "Error", 0.0, self._error_markdown("Model Unavailable", model_error), "{}"

        try:
            analyzer = self.registry.get_sentiment(model)
            result = analyzer.analyze(text)
            if result.error:
                return (
                    "Error",
                    0.0,
                    self._error_markdown("Sentiment Analysis Failed", result.error.message),
                    "{}",
                )
            payload = self._safe_json(result.model_dump())
            return result.label, result.confidence, result.explanation, payload
        except Exception as exc:  # pragma: no cover - defensive UI handler
            logger.exception("Sentiment handler failed")
            return "Error", 0.0, self._error_markdown("Unexpected Failure", str(exc)), "{}"

    def handle_summary(self, text: str, model: str) -> tuple[str, str, str]:
        """Summarization tab callback."""

        model_error = self._ensure_model_available(model)
        if model_error:
            return "", self._error_markdown("Model Unavailable", model_error), "{}"

        try:
            summarizer = self.registry.get_summarizer(model)
            result = summarizer.summarize(text)
            if result.error:
                return "", self._error_markdown("Summarization Failed", result.error.message), "{}"

            points = (
                "\n".join(f"- {point}" for point in result.key_points) or "- No key points returned"
            )
            payload = self._safe_json(result.model_dump())
            return result.summary, points, payload
        except Exception as exc:  # pragma: no cover - defensive UI handler
            logger.exception("Summary handler failed")
            return "", self._error_markdown("Unexpected Failure", str(exc)), "{}"

    def handle_translation(
        self, text: str, source_lang: str, target_lang: str, model: str
    ) -> tuple[str, str]:
        """Translation tab callback."""

        model_error = self._ensure_model_available(model)
        if model_error:
            return self._error_markdown("Model Unavailable", model_error), "{}"

        try:
            translator = self.registry.get_translator(model)
            result = translator.translate(text, source_lang, target_lang)
            if result.error:
                return self._error_markdown("Translation Failed", result.error.message), "{}"
            return result.translated_text, self._safe_json(result.model_dump())
        except Exception as exc:  # pragma: no cover - defensive UI handler
            logger.exception("Translation handler failed")
            return self._error_markdown("Unexpected Failure", str(exc)), "{}"

    def handle_chat(
        self,
        user_message: str,
        history_by_model: dict[str, list[dict[str, str]]] | None,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, list[dict[str, str]], dict[str, list[dict[str, str]]], str]:
        """Chat tab callback with per-model persistent state."""

        model_error = self._ensure_model_available(model)
        history_map = self._normalize_history_state(history_by_model)
        if model_error:
            message = {
                "role": "assistant",
                "content": self._error_markdown("Model Unavailable", model_error),
            }
            current = history_map.get(model, []) + [
                {"role": "user", "content": user_message},
                message,
            ]
            history_map[model] = current
            return "", current, history_map, "{}"

        try:
            engine = self.registry.get_chat(model)
            engine.reset()
            engine.load_history(history_map.get(model, []))
            result = engine.send(user_message, temperature=temperature, max_tokens=max_tokens)
            if result.error:
                message = {
                    "role": "assistant",
                    "content": self._error_markdown("Chat Failed", result.error.message),
                }
                updated_history = engine.history() + [message]
            else:
                updated_history = engine.history()

            history_map[model] = updated_history
            return "", updated_history, history_map, self._safe_json(result.model_dump())
        except Exception as exc:  # pragma: no cover - defensive UI handler
            logger.exception("Chat handler failed")
            message = {
                "role": "assistant",
                "content": self._error_markdown("Unexpected Failure", str(exc)),
            }
            current = history_map.get(model, []) + [
                {"role": "user", "content": user_message},
                message,
            ]
            history_map[model] = current
            return "", current, history_map, "{}"

    def switch_chat_model(
        self,
        history_by_model: dict[str, list[dict[str, str]]] | None,
        model: str,
    ) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
        """Load chat history for selected model when dropdown changes."""

        history_map = self._normalize_history_state(history_by_model)
        return history_map.get(model, []), history_map

    def reset_chat(
        self,
        model: str,
        history_by_model: dict[str, list[dict[str, str]]] | None,
    ) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], str]:
        """Clear selected model chat memory and state."""

        history_map = self._normalize_history_state(history_by_model)
        engine = self.registry.get_chat(model)
        engine.reset()
        history_map[model] = []
        return [], history_map, "{}"

    def handle_document_analysis(
        self,
        file_path: str | None,
        question: str,
        ocr_model: str,
        qa_model: str,
    ) -> tuple[str, str, str, str]:
        """Document tab callback for extraction, summary, and Q&A."""

        if not file_path:
            err = self._error_markdown("Input Required", "Upload PDF or image file first.")
            return err, "", "", "{}"

        ocr_model_error = self._ensure_model_available(ocr_model)
        qa_model_error = self._ensure_model_available(qa_model)
        if ocr_model_error or qa_model_error:
            err = self._error_markdown(
                "Model Unavailable",
                "\n".join(error for error in [ocr_model_error, qa_model_error] if error),
            )
            return err, "", "", "{}"

        try:
            analyzer = self.registry.get_document_analyzer(ocr_model=ocr_model, qa_model=qa_model)
            result = analyzer.analyze_document(file_path=file_path, question=question)
            payload = self._safe_json(result.model_dump())
            if result.error:
                err = self._error_markdown("Document Analysis Failed", result.error.message)
                return err, "", "", payload

            warning_block = ""
            if result.warnings:
                warning_lines = "\n".join(f"- {warning}" for warning in result.warnings[:5])
                warning_block = f"\n\n**Warnings**\n{warning_lines}"

            details = (
                f"Pages processed: **{result.pages_processed}**  \n"
                f"OCR model used: **{result.ocr_model_used}**  \n"
                f"QA model: **{result.qa_model}**  \n"
                f"Latency: **{result.latency_ms:.2f} ms**"
                f"{warning_block}"
            )
            return result.extracted_text, result.summary, result.answer + "\n\n" + details, payload
        except Exception as exc:  # pragma: no cover - defensive UI handler
            logger.exception("Document handler failed")
            err = self._error_markdown("Unexpected Failure", str(exc))
            return err, "", "", "{}"

    def run_benchmarks(
        self, prompt_profile: str, runs: int
    ) -> tuple[pd.DataFrame, str, str, str, str, str, str]:
        """Run benchmark suite and return table + chart artifacts for UI."""

        runner = BenchmarkRunner()
        visualizer = BenchmarkVisualizer()

        try:
            resolved_runs = max(1, int(runs))
            short_results = runner.run_all(
                prompt_key="short", runs=resolved_runs, models=BENCHMARK_MODELS
            )
            medium_results = runner.run_all(
                prompt_key="medium", runs=resolved_runs, models=BENCHMARK_MODELS
            )
            long_results = runner.run_all(
                prompt_key="long", runs=resolved_runs, models=BENCHMARK_MODELS
            )

            results_by_prompt = {
                "short": short_results,
                "medium": medium_results,
                "long": long_results,
            }
            runner.export_bundle(results_by_prompt=results_by_prompt, primary_prompt=prompt_profile)

            primary_results = results_by_prompt[prompt_profile]
            benchmark_df = pd.DataFrame([item.model_dump() for item in primary_results])
            benchmark_table = format_benchmark_table(primary_results)

            figures = visualizer.generate_all(
                medium_results=medium_results,
                short_results=short_results,
                long_results=long_results,
            )

            return (
                benchmark_df,
                benchmark_table,
                figures.get("latency", ""),
                figures.get("throughput", ""),
                figures.get("memory", ""),
                figures.get("prompt_scale", ""),
                figures.get("radar", ""),
            )
        finally:
            runner.close()

    def warmup_selected_models(self, selected_models: list[str]) -> str:
        """Warm selected models to reduce first-request latency."""

        if not selected_models:
            return "Select at least one model to warm."

        client = OllamaClient()
        try:
            rows = []
            for model in selected_models:
                status = client.warmup_model(model)
                icon = "OK" if status["ready"] else "FAIL"
                rows.append(
                    f"- {icon} `{model}` ({status['latency_ms']:.2f} ms): {status['message']}"
                )
            return "\n".join(rows)
        finally:
            client.close()

    def get_system_status(self) -> str:
        """Return runtime status for installed models and app configuration."""

        cfg = get_config()
        installed_models = self._refresh_model_cache(force=True)

        required_models = sorted(
            set(
                SENTIMENT_MODELS
                + SUMMARY_MODELS
                + [cfg.translation_model]
                + CHAT_MODELS
                + OCR_MODELS
                + BENCHMARK_MODELS
            )
        )

        lines = [
            "### Runtime Status",
            f"- Ollama base URL: `{cfg.ollama_base_url}`",
            f"- Required models: {len(required_models)}",
            f"- Installed models visible to Ollama: {len(installed_models)}",
            "",
            "#### Required Model Availability",
        ]

        for model in required_models:
            marker = "OK" if model in installed_models else "MISSING"
            lines.append(f"- {marker} `{model}`")

        return "\n".join(lines)


__all__ = [
    "AppHandlers",
    "BENCHMARK_MODELS",
    "CHAT_MODELS",
    "OCR_MODELS",
    "SENTIMENT_MODELS",
    "SUMMARY_MODELS",
    "SUPPORTED_LANGUAGES",
]
