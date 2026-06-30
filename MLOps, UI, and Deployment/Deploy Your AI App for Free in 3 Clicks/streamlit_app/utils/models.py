"""Model utilities for the Streamlit deployment application.

Inference strategy for each task:
1. Hugging Face Inference API (cloud-friendly primary)
2. Local Ollama model (local development fallback)
3. Deterministic rule-based fallback (always available)
"""

from __future__ import annotations

from collections import Counter
import copy
import json
import logging
import re
import time
from functools import lru_cache
from typing import Optional

import requests

from streamlit_app.utils.config import get_settings
from streamlit_app.utils.helpers import safe_json_parse

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"name": "English", "flag": "🇬🇧"},
    "fr": {"name": "French", "flag": "🇫🇷"},
    "de": {"name": "German", "flag": "🇩🇪"},
    "es": {"name": "Spanish", "flag": "🇪🇸"},
    "it": {"name": "Italian", "flag": "🇮🇹"},
    "pt": {"name": "Portuguese", "flag": "🇵🇹"},
    "nl": {"name": "Dutch", "flag": "🇳🇱"},
    "ro": {"name": "Romanian", "flag": "🇷🇴"},
    "ru": {"name": "Russian", "flag": "🇷🇺"},
}


def _registry() -> dict[str, str]:
    settings = get_settings()
    return {
        "sentiment_hf": "distilbert-base-uncased-finetuned-sst-2-english",
        "sentiment_local": settings.models.sentiment,
        "sentiment_local_alt": settings.models.sentiment_alt,
        "summarization_hf": "facebook/bart-large-cnn",
        "summarization_local": settings.models.summarization,
        "classification_hf": "facebook/bart-large-mnli",
        "classification_local": settings.models.classification,
        "translation_local": settings.models.translation,
    }


_INFERENCE_STATS: dict[str, dict[str, object]] = {
    "sentiment": {"requests": 0, "methods": Counter()},
    "summarization": {"requests": 0, "methods": Counter()},
    "classification": {"requests": 0, "methods": Counter()},
    "translation": {"requests": 0, "methods": Counter()},
}


def get_runtime_stats() -> dict[str, dict[str, object]]:
    """Return in-process inference counters for observability."""
    return {
        task: {
            "requests": int(values["requests"]),
            "methods": dict(values["methods"]),
        }
        for task, values in _INFERENCE_STATS.items()
    }


def reset_runtime_stats() -> None:
    """Reset in-process inference counters."""
    for task in _INFERENCE_STATS:
        _INFERENCE_STATS[task]["requests"] = 0
        _INFERENCE_STATS[task]["methods"] = Counter()


def clear_model_caches() -> None:
    """Clear in-process LRU caches for all task handlers."""
    _analyze_sentiment_cached.cache_clear()
    _summarize_text_cached.cache_clear()
    _classify_text_cached.cache_clear()
    _translate_text_cached.cache_clear()


def _record_inference(task: str, method: str) -> None:
    data = _INFERENCE_STATS[task]
    data["requests"] = int(data["requests"]) + 1
    methods: Counter = data["methods"]  # type: ignore[assignment]
    methods[method] += 1


def _elapsed_seconds(start: float) -> float:
    return round(max(0.0, time.perf_counter() - start), 4)


def _call_hf_inference_api(
    model: str,
    inputs: str,
    task: str = "text-classification",
    parameters: Optional[dict] = None,
) -> Optional[dict | list]:
    """Call Hugging Face Inference API and return parsed JSON response."""
    token = get_settings().hf_api_token
    if not token:
        return None

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict[str, object] = {"inputs": inputs}
    if parameters:
        payload["parameters"] = parameters

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=35)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("HF Inference API failed (%s, task=%s): %s", model, task, exc)
        return None


def _call_ollama_api(model: str, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
    """Call local Ollama generate endpoint and return model text output."""
    base_url = get_settings().ollama_base_url.rstrip("/")
    url = f"{base_url}/api/generate"

    payload: dict[str, object] = {"model": model, "prompt": prompt, "stream": False}
    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        body = response.json()
        return str(body.get("response", ""))
    except requests.exceptions.RequestException as exc:
        logger.warning("Ollama call failed (%s): %s", model, exc)
        return None


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------
def analyze_sentiment_hf(text: str) -> list[dict[str, object]]:
    """Sentiment analysis using Hugging Face Inference API."""
    result = _call_hf_inference_api(_registry()["sentiment_hf"], text)
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def analyze_sentiment_ollama(text: str, model: Optional[str] = None) -> str:
    """Sentiment analysis using local Ollama model."""
    selected_model = model or _registry()["sentiment_local"]
    system = (
        "You are a sentiment analysis engine. "
        "Analyze the sentiment of the given text and respond with ONLY "
        "a JSON object with keys: 'label' (POSITIVE, NEGATIVE, or NEUTRAL) "
        "and 'score' (a float between 0 and 1)."
    )
    return _call_ollama_api(selected_model, text, system) or ""


def analyze_sentiment_rule_based(text: str) -> dict[str, object]:
    """Deterministic keyword-based sentiment fallback."""
    positive_words = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "happy", "love", "beautiful", "awesome", "best", "brilliant",
        "outstanding", "superb", "perfect", "joy", "delightful", "positive",
        "nice", "helpful", "enjoy", "pleased", "impressed", "splendid",
        "magnificent", "terrific", "glad", "thankful", "grateful", "friendly",
        "kind", "generous", "caring", "thoughtful", "warm", "welcoming",
        "lovely", "charming", "pleasant", "satisfied",
    }
    negative_words = {
        "bad", "terrible", "awful", "horrible", "worst", "hate", "poor",
        "dreadful", "nasty", "evil", "horrendous", "atrocious", "abysmal",
        "lousy", "pathetic", "miserable", "sad", "angry", "furious",
        "annoying", "irritating", "disappointing", "frustrating", "depressing",
        "painful", "horrific", "tragic", "unpleasant", "toxic", "hostile",
        "cruel", "ruthless", "vile", "hideous", "grim", "dire", "dismal",
        "inferior",
    }

    words = set(re.findall(r"\w+", text.lower()))
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    total = pos_count + neg_count

    if total == 0:
        return {"label": "NEUTRAL", "score": 0.5}
    if pos_count > neg_count:
        return {"label": "POSITIVE", "score": round(pos_count / total, 4)}
    if neg_count > pos_count:
        return {"label": "NEGATIVE", "score": round(neg_count / total, 4)}
    return {"label": "NEUTRAL", "score": 0.5}


def _analyze_sentiment_impl(text: str) -> dict[str, object]:
    start = time.perf_counter()

    if not text.strip():
        method = "Rule-based (fallback)"
        _record_inference("sentiment", method)
        return {
            "label": "NEUTRAL",
            "confidence": 0.5,
            "scores": {"NEUTRAL": 0.5},
            "method": method,
            "inference_time": 0.0,
        }

    hf_result = analyze_sentiment_hf(text)
    if hf_result:
        top = hf_result[0]
        label = str(top.get("label", "NEUTRAL")).upper()
        confidence = float(top.get("score", 0.5))
        scores = {
            str(item.get("label", "UNKNOWN")).upper(): round(float(item.get("score", 0.0)), 4)
            for item in hf_result[:5]
        }
        method = "Hugging Face Inference API"
        _record_inference("sentiment", method)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": scores,
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    ollama_payload = analyze_sentiment_ollama(text, model=_registry()["sentiment_local"])
    parsed = safe_json_parse(ollama_payload) if ollama_payload else None
    if isinstance(parsed, dict):
        label = str(parsed.get("label", "NEUTRAL")).upper()
        confidence = float(parsed.get("score", 0.5))
        method = f"Ollama ({_registry()['sentiment_local']})"
        _record_inference("sentiment", method)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": {label: round(confidence, 4)},
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    # Try alternative local model before deterministic fallback.
    alt_payload = analyze_sentiment_ollama(text, model=_registry()["sentiment_local_alt"])
    alt_parsed = safe_json_parse(alt_payload) if alt_payload else None
    if isinstance(alt_parsed, dict):
        label = str(alt_parsed.get("label", "NEUTRAL")).upper()
        confidence = float(alt_parsed.get("score", 0.5))
        method = f"Ollama ({_registry()['sentiment_local_alt']})"
        _record_inference("sentiment", method)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": {label: round(confidence, 4)},
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    fallback = analyze_sentiment_rule_based(text)
    confidence = float(fallback["score"])
    label = str(fallback["label"])
    method = "Rule-based (fallback)"
    _record_inference("sentiment", method)
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {label: round(confidence, 4)},
        "method": method,
        "inference_time": _elapsed_seconds(start),
    }


@lru_cache(maxsize=1024)
def _analyze_sentiment_cached(text: str) -> dict[str, object]:
    return _analyze_sentiment_impl(text)


def analyze_sentiment(text: str, use_cache: bool = True) -> dict[str, object]:
    """Public sentiment entrypoint with in-process caching."""
    result = _analyze_sentiment_cached(text) if use_cache else _analyze_sentiment_impl(text)
    return copy.deepcopy(result)


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def summarize_text_hf(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Summarize text using Hugging Face hosted model."""
    params = {
        "max_length": max(20, int(max_length)),
        "min_length": max(5, min(int(min_length), int(max_length) - 1)),
    }
    result = _call_hf_inference_api(
        _registry()["summarization_hf"],
        text,
        task="summarization",
        parameters=params,
    )
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return str(result[0].get("summary_text", ""))
    return ""


def summarize_text_ollama(text: str, model: Optional[str] = None) -> str:
    """Summarize text using local Ollama model."""
    selected_model = model or _registry()["summarization_local"]
    system = (
        "You are a summarization engine. Summarize the text concisely "
        "while preserving key information. Output ONLY the summary."
    )
    return _call_ollama_api(selected_model, f"Summarize:\n{text}", system) or ""


def summarize_text_fallback(text: str) -> str:
    """Simple extractive summarization fallback."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) <= 2:
        return text

    words = re.findall(r"\w+", text.lower())
    freq = Counter(words)

    def score(sentence: str) -> float:
        sent_words = re.findall(r"\w+", sentence.lower())
        if not sent_words:
            return 0.0
        return sum(freq[word] for word in sent_words) / len(sent_words)

    ranked = sorted(((sent, score(sent)) for sent in sentences), key=lambda item: item[1], reverse=True)
    top = {sent for sent, _ in ranked[: max(1, len(sentences) // 3)]}
    ordered = [sent for sent in sentences if sent in top]
    return " ".join(ordered) if ordered else sentences[0]


def _clip_to_word_count(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""

    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def _summarize_text_impl(text: str, max_length: int) -> dict[str, object]:
    start = time.perf_counter()

    if len(text.strip()) < 50:
        method = "Input too short, returned as-is"
        _record_inference("summarization", method)
        return {
            "summary": text,
            "method": method,
            "inference_time": 0.0,
        }

    hf = summarize_text_hf(text, max_length=max_length)
    if hf:
        method = "Hugging Face Inference API (bart-large-cnn)"
        _record_inference("summarization", method)
        return {
            "summary": _clip_to_word_count(hf, max_length),
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    ollama = summarize_text_ollama(text)
    if ollama:
        method = f"Ollama ({_registry()['summarization_local']})"
        _record_inference("summarization", method)
        return {
            "summary": _clip_to_word_count(ollama, max_length),
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    fallback = summarize_text_fallback(text)
    method = "Extractive fallback (word frequency)"
    _record_inference("summarization", method)
    return {
        "summary": _clip_to_word_count(fallback, max_length),
        "method": method,
        "inference_time": _elapsed_seconds(start),
    }


@lru_cache(maxsize=512)
def _summarize_text_cached(text: str, max_length: int) -> dict[str, object]:
    return _summarize_text_impl(text, max_length=max_length)


def summarize_text(text: str, max_length: int = 130, use_cache: bool = True) -> dict[str, object]:
    """Public summarization entrypoint with in-process caching."""
    max_words = max(10, int(max_length))
    result = (
        _summarize_text_cached(text, max_words)
        if use_cache
        else _summarize_text_impl(text, max_length=max_words)
    )
    return copy.deepcopy(result)


# ---------------------------------------------------------------------------
# Text Classification
# ---------------------------------------------------------------------------
def classify_text_hf(text: str, labels: list[str]) -> list[dict[str, object]]:
    """Zero-shot text classification using Hugging Face Inference API."""
    result = _call_hf_inference_api(
        _registry()["classification_hf"],
        text,
        task="zero-shot-classification",
        parameters={"candidate_labels": labels, "multi_label": False},
    )

    if isinstance(result, dict):
        output_labels = result.get("labels", [])
        output_scores = result.get("scores", [])
        return [
            {"label": str(label), "score": round(float(score), 4)}
            for label, score in zip(output_labels, output_scores)
        ]

    if isinstance(result, list):
        parsed: list[dict[str, object]] = []
        for item in result:
            if isinstance(item, dict) and "label" in item and "score" in item:
                parsed.append(
                    {
                        "label": str(item["label"]),
                        "score": round(float(item["score"]), 4),
                    }
                )
        return parsed

    return []


def classify_text_ollama(text: str, labels: list[str], model: Optional[str] = None) -> str:
    """Text classification using local Ollama model."""
    selected_model = model or _registry()["classification_local"]
    system = (
        "You are a text classification engine. "
        f"Classify the text into one of these categories: {', '.join(labels)}. "
        "Output ONLY a JSON object with keys 'label' and 'score'."
    )
    prompt = f"Text: {text}\nCategories: {', '.join(labels)}"
    return _call_ollama_api(selected_model, prompt, system) or ""


def classify_text_fallback(text: str, labels: list[str]) -> list[dict[str, object]]:
    """Keyword-based classification fallback."""
    words = set(re.findall(r"\w+", text.lower()))

    keyword_map: dict[str, set[str]] = {
        "technology": {
            "computer", "software", "tech", "digital", "code", "programming", "data", "algorithm",
            "internet", "web", "app", "device", "electronic", "cyber", "robot", "ai", "machine",
            "cloud", "network", "system", "database",
        },
        "business": {
            "company", "market", "revenue", "profit", "startup", "enterprise", "corporate",
            "management", "strategy", "investment", "finance", "economic", "industry", "product",
            "service", "customer", "sales", "growth", "funding", "acquisition", "partnership",
        },
        "science": {
            "research", "study", "experiment", "theory", "hypothesis", "analysis", "laboratory",
            "scientific", "discovery", "biology", "chemistry", "physics", "genetic", "molecule",
            "organism", "evolution", "species", "ecosystem", "climate", "energy", "particle",
        },
        "health": {
            "health", "medical", "disease", "treatment", "patient", "doctor", "hospital", "medicine",
            "drug", "therapy", "surgery", "clinical", "diagnosis", "symptom", "wellness", "fitness",
            "nutrition", "mental", "exercise", "diet", "recovery", "prevention", "vaccine", "care",
        },
        "education": {
            "education", "learning", "student", "teacher", "school", "university", "college", "course",
            "training", "knowledge", "skill", "curriculum", "classroom", "lesson", "study", "academic",
            "degree", "scholarship", "exam", "grade",
        },
        "entertainment": {
            "movie", "music", "game", "film", "show", "artist", "entertainment", "concert", "theater",
            "performance", "audience", "actor", "director", "album", "song", "dance", "comedy",
            "drama", "series", "streaming", "fan", "celebrity", "festival",
        },
        "sports": {
            "sport", "sports", "football", "soccer", "basketball", "baseball", "tennis", "match",
            "player", "team", "coach", "league", "tournament", "score", "goal", "athlete", "running",
        },
    }

    raw_scores: dict[str, float] = {}
    for label in labels:
        keywords = keyword_map.get(label.lower(), set())
        if not keywords:
            raw_scores[label] = 0.0
            continue
        matches = len(words & keywords)
        raw_scores[label] = float(matches)

    total = sum(raw_scores.values())
    if total <= 0:
        return [{"label": label, "score": 0.0} for label in labels]

    normalized = [
        {"label": label, "score": round(score / total, 4)}
        for label, score in raw_scores.items()
    ]
    return sorted(normalized, key=lambda item: item["score"], reverse=True)


def _classify_text_impl(text: str, labels: tuple[str, ...]) -> dict[str, object]:
    start = time.perf_counter()
    labels_list = list(labels)

    if not text.strip():
        winner = labels_list[0]
        method = "Empty input fallback"
        _record_inference("classification", method)
        return {
            "label": winner,
            "winner": winner,
            "confidence": 0.0,
            "scores": {label: 0.0 for label in labels_list},
            "method": method,
            "inference_time": 0.0,
        }

    hf = classify_text_hf(text, labels_list)
    if hf:
        top = hf[0]
        winner = str(top["label"])
        confidence = float(top["score"])
        scores = {str(item["label"]): float(item["score"]) for item in hf}
        method = "Hugging Face Inference API (bart-large-mnli)"
        _record_inference("classification", method)
        return {
            "label": winner,
            "winner": winner,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    ollama_payload = classify_text_ollama(text, labels_list)
    parsed = safe_json_parse(ollama_payload) if ollama_payload else None
    if isinstance(parsed, dict):
        winner = str(parsed.get("label", labels_list[0]))
        if winner not in labels_list:
            winner = labels_list[0]
        confidence = float(parsed.get("score", 0.5))
        scores = {label: (round(confidence, 4) if label == winner else 0.0) for label in labels_list}
        method = f"Ollama ({_registry()['classification_local']})"
        _record_inference("classification", method)
        return {
            "label": winner,
            "winner": winner,
            "confidence": round(confidence, 4),
            "scores": scores,
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    fallback = classify_text_fallback(text, labels_list)
    winner = str(fallback[0]["label"]) if fallback else labels_list[0]
    confidence = float(fallback[0]["score"]) if fallback else 0.0
    scores = {str(item["label"]): float(item["score"]) for item in fallback}
    method = "Keyword-based fallback"
    _record_inference("classification", method)
    return {
        "label": winner,
        "winner": winner,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "method": method,
        "inference_time": _elapsed_seconds(start),
    }


@lru_cache(maxsize=1024)
def _classify_text_cached(text: str, labels: tuple[str, ...]) -> dict[str, object]:
    return _classify_text_impl(text, labels)


def classify_text(text: str, labels: Optional[list[str]] = None, use_cache: bool = True) -> dict[str, object]:
    """Public classification entrypoint with in-process caching."""
    if labels is None:
        labels = ["technology", "business", "science", "health", "education", "entertainment"]

    clean_labels = [label.strip() for label in labels if label and label.strip()]
    if not clean_labels:
        raise IndexError("labels list cannot be empty")

    labels_key = tuple(clean_labels)
    result = _classify_text_cached(text, labels_key) if use_cache else _classify_text_impl(text, labels_key)
    return copy.deepcopy(result)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def _hf_translation_model(source_lang: str, target_lang: str) -> Optional[str]:
    source = source_lang.lower()
    target = target_lang.lower()
    if source == target:
        return None

    en_out = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "de": "Helsinki-NLP/opus-mt-en-de",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "it": "Helsinki-NLP/opus-mt-en-it",
        "pt": "Helsinki-NLP/opus-mt-en-tc-big-pt",
        "nl": "Helsinki-NLP/opus-mt-en-nl",
        "ro": "Helsinki-NLP/opus-mt-en-ro",
        "ru": "Helsinki-NLP/opus-mt-en-ru",
    }

    if source == "en":
        return en_out.get(target)

    # Try common reverse pairs back to English.
    if target == "en" and source in SUPPORTED_LANGUAGES:
        return f"Helsinki-NLP/opus-mt-{source}-en"

    return None


def translate_text_hf(text: str, target_lang: str = "fr", source_lang: str = "en") -> str:
    """Translate text via Hugging Face when model pair is available."""
    model = _hf_translation_model(source_lang, target_lang)
    if not model:
        return ""

    result = _call_hf_inference_api(model, text, task="translation")
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return str(result[0].get("translation_text", ""))
    return ""


def translate_text_ollama(
    text: str,
    target_lang: str = "French",
    source_lang: str = "English",
    model: Optional[str] = None,
) -> str:
    """Translate text via local Ollama model."""
    selected_model = model or _registry()["translation_local"]
    system = (
        f"You are a professional {source_lang} to {target_lang} translator. "
        "Produce only the translation and no additional commentary."
    )
    prompt = f"Please translate the following text:\n{text}"
    return _call_ollama_api(selected_model, prompt, system) or ""


def translate_text_fallback(text: str, target_lang: str, source_lang: str = "en") -> str:
    """Deterministic fallback message when translation providers are unavailable."""
    source_name = SUPPORTED_LANGUAGES.get(source_lang, {"name": source_lang}).get("name", source_lang)
    target_name = SUPPORTED_LANGUAGES.get(target_lang, {"name": target_lang}).get("name", target_lang)
    return (
        f"[Translation from {source_name} to {target_name} requires either "
        "a Hugging Face API token (HF_API_TOKEN) or a local Ollama instance.]"
    )


def _translate_text_impl(text: str, source_lang: str, target_lang: str) -> dict[str, object]:
    start = time.perf_counter()

    if source_lang == target_lang:
        method = "No-op (source equals target)"
        _record_inference("translation", method)
        return {
            "translated_text": text,
            "method": method,
            "inference_time": 0.0,
        }

    if not text.strip():
        method = "Not available (configure API or Ollama)"
        _record_inference("translation", method)
        return {
            "translated_text": translate_text_fallback(text, target_lang, source_lang),
            "method": method,
            "inference_time": 0.0,
        }

    hf = translate_text_hf(text, target_lang=target_lang, source_lang=source_lang)
    if hf:
        method = f"Hugging Face Inference API ({source_lang}->{target_lang})"
        _record_inference("translation", method)
        return {
            "translated_text": hf,
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    source_name = SUPPORTED_LANGUAGES.get(source_lang, {"name": source_lang}).get("name", source_lang)
    target_name = SUPPORTED_LANGUAGES.get(target_lang, {"name": target_lang}).get("name", target_lang)
    ollama = translate_text_ollama(text, target_lang=target_name, source_lang=source_name)
    if ollama:
        method = f"Ollama ({_registry()['translation_local']})"
        _record_inference("translation", method)
        return {
            "translated_text": ollama,
            "method": method,
            "inference_time": _elapsed_seconds(start),
        }

    method = "Not available (configure API or Ollama)"
    _record_inference("translation", method)
    return {
        "translated_text": translate_text_fallback(text, target_lang, source_lang),
        "method": method,
        "inference_time": _elapsed_seconds(start),
    }


@lru_cache(maxsize=1024)
def _translate_text_cached(text: str, source_lang: str, target_lang: str) -> dict[str, object]:
    return _translate_text_impl(text, source_lang, target_lang)


def translate_text(
    text: str,
    source_lang: Optional[str] = None,
    target_lang: str = "fr",
    use_cache: bool = True,
) -> dict[str, object]:
    """Public translation entrypoint with in-process caching."""
    source = (source_lang or "en").lower()
    target = target_lang.lower()

    result = (
        _translate_text_cached(text, source, target)
        if use_cache
        else _translate_text_impl(text, source, target)
    )
    return copy.deepcopy(result)
