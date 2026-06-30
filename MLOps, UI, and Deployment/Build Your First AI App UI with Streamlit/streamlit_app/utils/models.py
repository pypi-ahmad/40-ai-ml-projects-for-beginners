"""Model integration utilities for local Ollama inference.

This file intentionally exposes small functional wrappers (easier for beginners)
while centralizing robust behavior: prompt templates, error handling, parsing,
and benchmark metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
import io
import json
import logging
import re
import statistics
import time
from typing import Any

import ollama

from streamlit_app.config import APP_CONFIG
from streamlit_app.services.prompts import (
    benchmark_quality_prompt,
    chat_system_prompt,
    classification_system_prompt,
    ocr_analysis_system_prompt,
    sentiment_system_prompt,
    summarization_system_prompt,
    translation_system_prompt,
)
from streamlit_app.services.schemas import BenchmarkRecord, BenchmarkSummary, ClassificationResult, SentimentResult


logger = logging.getLogger(__name__)

# Backward-compatible constants used by tests and pages.
MODEL_SENTIMENT = APP_CONFIG.models.sentiment
MODEL_SUMMARY = APP_CONFIG.models.summarization
MODEL_CLASSIFY = APP_CONFIG.models.classification
MODEL_TRANSLATE = APP_CONFIG.models.translation
MODEL_CHAT = APP_CONFIG.models.chat
MODEL_OCR = APP_CONFIG.models.ocr_primary
MODEL_OCR_FALLBACK = APP_CONFIG.models.ocr_fallback
MODEL_EMBED = APP_CONFIG.models.embedding
MODEL_FAST = APP_CONFIG.models.chat_fast
MODEL_NEMOTRON = APP_CONFIG.models.benchmark_extra

BENCHMARK_MODELS = [
    (MODEL_FAST, MODEL_FAST),
    (MODEL_CHAT, MODEL_CHAT),
    (MODEL_SENTIMENT, MODEL_SENTIMENT),
    (MODEL_NEMOTRON, MODEL_NEMOTRON),
]


class OllamaClientError(RuntimeError):
    """Raised when Ollama call fails and caller needs fallback behavior."""


def is_ollama_available() -> bool:
    """Return True when Ollama API is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def list_available_models() -> list[str]:
    """Return local model names from Ollama daemon."""
    try:
        data = ollama.list()
    except Exception:
        return []

    names: list[str] = []
    for item in data.get("models", []):
        model_name = item.get("model") or item.get("name")
        if model_name:
            names.append(str(model_name))
    return names


def _current_memory_mb() -> float:
    """Return current process memory in MB.

    Uses Linux ru_maxrss semantics (KB). Works well for comparative local benchmarks.
    """
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / 1024.0
    except Exception:
        return 0.0


def _safe_json_extract(raw: str) -> dict[str, Any] | None:
    """Extract first JSON object from raw model output."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _safe_float(value: Any, fallback: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, val))


def _quality_proxy(text: str) -> float:
    """Simple quality proxy for benchmark comparison.

    This is not a replacement for human eval. It helps compare response diversity
    and verbosity in automated benchmark runs.
    """
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not tokens:
        return 0.0
    lexical_diversity = len(set(tokens)) / len(tokens)
    length_factor = min(len(tokens) / 120.0, 1.0)
    return round((0.6 * lexical_diversity) + (0.4 * length_factor), 4)


def _call_ollama(
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> str:
    """Send a non-streaming chat request to Ollama and return model text.

    This function is intentionally kept as simple wrapper so tests can patch it.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens if max_tokens is not None else APP_CONFIG.default_max_tokens,
            },
        )
    except Exception as exc:
        logger.error("Ollama call failed for model %s: %s", model, exc)
        return f"Error: {exc}"

    content = response.get("message", {}).get("content", "")
    return str(content).strip()


def analyze_sentiment(text: str, model: str = MODEL_SENTIMENT) -> dict[str, Any]:
    """Analyze sentiment and return structured result dictionary."""
    raw = _call_ollama(model, text, sentiment_system_prompt())
    parsed = _safe_json_extract(raw)
    if not parsed:
        return SentimentResult(
            sentiment="neutral",
            confidence=0.0,
            explanation="Could not parse model output.",
        ).to_dict()

    result = SentimentResult(
        sentiment=str(parsed.get("sentiment", "neutral")).lower(),
        confidence=_safe_float(parsed.get("confidence"), 0.0),
        explanation=str(parsed.get("explanation", "No explanation returned.")),
    )
    if result.sentiment not in {"positive", "negative", "neutral"}:
        result.sentiment = "neutral"
    return result.to_dict()


def summarize_text(
    text: str,
    max_length: int = 150,
    model: str = MODEL_SUMMARY,
    temperature: float = 0.2,
) -> str:
    """Generate abstractive summary text."""
    return _call_ollama(
        model=model,
        prompt=text,
        system=summarization_system_prompt(max_length),
        temperature=temperature,
        max_tokens=max_length * 2,
    )


def classify_text(
    text: str,
    categories: list[str] | None = None,
    model: str = MODEL_CLASSIFY,
) -> dict[str, Any]:
    """Classify text into provided categories using zero-shot prompting."""
    active_categories = categories or [
        "Technology",
        "Business",
        "Finance",
        "Healthcare",
        "Education",
        "Politics",
        "Sports",
        "Other",
    ]
    raw = _call_ollama(model, text, classification_system_prompt(active_categories))
    parsed = _safe_json_extract(raw)
    if not parsed:
        return ClassificationResult(
            category="Other",
            confidence=0.0,
            reason="Could not classify model output.",
        ).to_dict()

    category = str(parsed.get("category", "Other"))
    if category not in active_categories:
        category = "Other"

    result = ClassificationResult(
        category=category,
        confidence=_safe_float(parsed.get("confidence"), 0.0),
        reason=str(parsed.get("reason", "No reason returned.")),
    )
    return result.to_dict()


def translate_text(
    text: str,
    target_lang: str = "French",
    model: str = MODEL_TRANSLATE,
    temperature: float = 0.0,
) -> str:
    """Translate text into target language."""
    return _call_ollama(
        model=model,
        prompt=text,
        system=translation_system_prompt(target_lang),
        temperature=temperature,
    )


def chat_response(
    messages: list[dict[str, str]],
    model: str = MODEL_CHAT,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    """Send chat history to Ollama and return assistant response."""
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens if max_tokens is not None else APP_CONFIG.default_max_tokens,
            },
        )
    except Exception as exc:
        logger.error("Chat request failed for model %s: %s", model, exc)
        return f"Error: {exc}"

    return str(response.get("message", {}).get("content", "")).strip()


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF using pdfplumber page-wise parsing."""
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text.strip())
    return "\n\n".join(text_parts)


def extract_text_from_docx(content: bytes) -> str:
    """Extract paragraphs from DOCX binary content."""
    from docx import Document

    doc = Document(io.BytesIO(content))
    parts = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    return "\n".join(parts)


def extract_text_from_image(content: bytes) -> str:
    """Extract text from image using pytesseract OCR."""
    from PIL import Image
    import pytesseract

    image = Image.open(io.BytesIO(content))
    return pytesseract.image_to_string(image)


def _run_ocr_llm_analysis(extracted_text: str, model: str) -> str:
    return _call_ollama(
        model=model,
        prompt=extracted_text[:7000],
        system=ocr_analysis_system_prompt(),
        temperature=0.1,
        max_tokens=700,
    )


def ocr_analyze(
    content: bytes,
    filename: str,
    primary_model: str = MODEL_OCR,
    fallback_model: str = MODEL_OCR_FALLBACK,
) -> str:
    """Run OCR/extraction + LLM analysis with fallback OCR model.

    Flow:
        1. Extract text from file based on extension.
        2. Analyze extracted text with primary OCR model.
        3. Fallback to secondary OCR model if primary fails.
    """
    suffix = filename.lower().split(".")[-1] if "." in filename else ""

    if suffix == "pdf":
        extracted_text = extract_text_from_pdf(content)
    elif suffix == "docx":
        extracted_text = extract_text_from_docx(content)
    elif suffix in {"png", "jpg", "jpeg", "bmp", "tiff"}:
        extracted_text = extract_text_from_image(content)
    else:
        return "Unsupported file type. Upload PDF, DOCX, PNG, JPG, JPEG, BMP, or TIFF."

    if not extracted_text.strip():
        return "No text could be extracted from the uploaded document."

    primary = _run_ocr_llm_analysis(extracted_text, primary_model)
    if not primary.startswith("Error:"):
        return primary

    fallback = _run_ocr_llm_analysis(extracted_text, fallback_model)
    if not fallback.startswith("Error:"):
        return fallback

    return (
        "Primary and fallback OCR analysis models failed. "
        f"Primary error: {primary}. Fallback error: {fallback}."
    )


def benchmark_inference(
    model: str,
    prompt: str,
    system: str | None = None,
    runs: int = 3,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Benchmark model inference latency, memory, throughput, and quality proxy."""
    latencies: list[float] = []
    outputs: list[str] = []
    records: list[BenchmarkRecord] = []

    for run_id in range(1, runs + 1):
        start_memory = _current_memory_mb()
        start = time.perf_counter()

        output = _call_ollama(
            model=model,
            prompt=prompt,
            system=system or benchmark_quality_prompt(),
            temperature=temperature,
        )

        elapsed = time.perf_counter() - start
        end_memory = _current_memory_mb()

        words = len(output.split())
        throughput = words / elapsed if elapsed > 0 else 0.0
        lexical_diversity = _quality_proxy(output)

        latencies.append(elapsed)
        outputs.append(output)
        records.append(
            BenchmarkRecord(
                model=model,
                run_id=run_id,
                latency_seconds=elapsed,
                output_length=len(output),
                output_word_count=words,
                throughput_words_per_second=throughput,
                process_memory_mb=max(start_memory, end_memory),
                lexical_diversity=lexical_diversity,
            )
        )

    summary = BenchmarkSummary(
        model=model,
        runs=runs,
        mean_latency=statistics.mean(latencies),
        median_latency=statistics.median(latencies),
        min_latency=min(latencies),
        max_latency=max(latencies),
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        mean_output_len=statistics.mean(record.output_length for record in records),
        mean_output_words=statistics.mean(record.output_word_count for record in records),
        mean_throughput_wps=statistics.mean(record.throughput_words_per_second for record in records),
        mean_memory_mb=statistics.mean(record.process_memory_mb for record in records),
        mean_quality_score=statistics.mean(record.lexical_diversity for record in records),
        created_at_utc=BenchmarkSummary.now_iso(),
    )

    summary_dict = summary.to_dict()
    # Backward-compatible fields expected by previous tests.
    summary_dict["outputs"] = outputs
    summary_dict["records"] = [record.to_dict() for record in records]
    return summary_dict


def run_benchmark_matrix(
    models: Sequence[str],
    prompt: str,
    runs: int,
    system_prompt: str | None = None,
    temperature: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run benchmark for multiple models and return summary + run-level rows."""
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for model in models:
        result = benchmark_inference(
            model=model,
            prompt=prompt,
            system=system_prompt,
            runs=runs,
            temperature=temperature,
        )
        summaries.append({key: value for key, value in result.items() if key not in {"outputs", "records"}})
        rows.extend(result["records"])

    return summaries, rows


def get_embedding(text: str, model: str = MODEL_EMBED) -> list[float]:
    """Return embedding vector for input text."""
    try:
        # Newer Ollama versions expose "embed", older versions use "embeddings".
        if hasattr(ollama, "embed"):
            response = ollama.embed(model=model, input=text)
            if isinstance(response, dict):
                if "embeddings" in response and response["embeddings"]:
                    return list(response["embeddings"][0])
                if "embedding" in response:
                    return list(response["embedding"])
        response = ollama.embeddings(model=model, prompt=text)
        return list(response.get("embedding", []))
    except Exception as exc:
        logger.error("Embedding request failed: %s", exc)
        return []


def _try_parse_json(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible JSON parser used by tests."""
    parsed = _safe_json_extract(raw)
    return parsed if parsed is not None else fallback


def build_default_chat_messages(user_prompt: str) -> list[dict[str, str]]:
    """Build initial chat payload with system prompt + user message."""
    return [
        {"role": "system", "content": chat_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]
