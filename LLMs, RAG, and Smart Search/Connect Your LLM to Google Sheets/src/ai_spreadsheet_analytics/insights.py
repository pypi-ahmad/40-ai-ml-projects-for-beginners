"""Hybrid deterministic + LLM insight generation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ai_spreadsheet_analytics.llm.base import LLMClient
from ai_spreadsheet_analytics.prompts import SYSTEM_PROMPT, build_prompt
from ai_spreadsheet_analytics.schemas import InsightPacket


class InsightEngine:
    """Generate business insights from deterministic analytics payloads."""

    def __init__(self, llm_client: LLMClient, default_temperature: float = 0.0) -> None:
        self.llm_client = llm_client
        self.default_temperature = default_temperature

    def generate(
        self,
        analytics_payload: dict[str, Any],
        role: str,
        model: str,
        max_words: int = 180,
    ) -> InsightPacket:
        """Generate one insight packet."""
        prompt = build_prompt(role=role, deterministic_payload=analytics_payload, max_words=max_words)
        response = asyncio.run(
            self.llm_client.agenerate(
                model=model,
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=self.default_temperature,
            )
        )
        findings = _extract_list_items(response.text, "key")
        recommendations = _extract_list_items(response.text, "recommend")
        return InsightPacket(
            prompt_role=role,
            model=model,
            summary=response.text,
            findings=findings,
            recommendations=recommendations,
            deterministic_evidence=analytics_payload,
            latency_ms=response.latency_ms,
            token_estimate=response.token_estimate,
        )

    def compare_models(
        self,
        analytics_payload: dict[str, Any],
        role: str,
        models: list[str],
        max_words: int = 180,
    ) -> list[InsightPacket]:
        """Generate insight outputs across multiple models."""
        return [
            self.generate(
                analytics_payload=analytics_payload,
                role=role,
                model=model,
                max_words=max_words,
            )
            for model in models
        ]


def _extract_list_items(text: str, marker: str) -> list[str]:
    marker_lower = marker.lower()
    candidates = [line.strip("- ").strip() for line in text.splitlines() if marker_lower in line.lower()]
    return candidates[:5]


def insight_packet_to_json(packet: InsightPacket) -> str:
    """Serialize insight packet to JSON string."""
    return json.dumps(
        {
            "prompt_role": packet.prompt_role,
            "model": packet.model,
            "summary": packet.summary,
            "findings": packet.findings,
            "recommendations": packet.recommendations,
            "latency_ms": packet.latency_ms,
            "token_estimate": packet.token_estimate,
        },
        indent=2,
    )
