"""Connector base classes and resilient HTTP execution."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from api_intel_agent.auth import AuthManager
from api_intel_agent.config import load_settings
from api_intel_agent.core.schemas import ConnectorResult, ErrorRecord


class ConnectorError(RuntimeError):
    pass


@dataclass(slots=True)
class APIRequest:
    endpoint: str
    method: str = "GET"
    params: dict[str, Any] | None = None
    json: dict[str, Any] | None = None


class BaseConnector:
    provider: str

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.settings = load_settings()
        api_cfg = self.settings.apis.get(provider, {})
        self.base_url: str = api_cfg.get("base_url", "")
        self.auth = AuthManager()
        self.semaphore = asyncio.Semaphore(self.settings.agent.max_parallel_calls)
        self._last_call_monotonic = 0.0
        self._min_interval_seconds = 0.2

    def prepare_auth(self) -> tuple[dict[str, str], str]:
        auth = self.auth.auth_headers(self.provider)
        return auth.headers, auth.status

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        return APIRequest(endpoint="/", params=params or {"q": query})

    def normalize(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("items", "results", "data", "articles", "posts"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        return []

    def validate(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return records

    def next_page(self, payload: Any, params: dict[str, Any]) -> dict[str, Any] | None:
        _ = payload
        _ = params
        return None

    async def _rate_limit(self) -> None:
        delta = time.monotonic() - self._last_call_monotonic
        wait = self._min_interval_seconds - delta
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call_monotonic = time.monotonic()

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> ConnectorResult:
        headers, auth_status = self.prepare_auth()
        if auth_status == "skipped_missing_credentials":
            return ConnectorResult(
                provider=self.provider,
                endpoint="/",
                status="skipped_missing_credentials",
                error=ErrorRecord(
                    code="MISSING_CREDENTIALS",
                    message="Missing credentials for provider",
                    provider=self.provider,
                    retryable=False,
                ),
            )

        request = self.build_request(query, params)
        all_records: list[dict[str, Any]] = []
        pagination: dict[str, Any] = {}
        latency_total = 0.0
        attempts = 0

        async with self.semaphore:
            current_params = dict(request.params or {})
            while True:
                attempts += 1
                try:
                    response, latency = await self._call_with_retry(request, current_params, headers)
                    latency_total += latency
                    payload = response.json()
                    records = self.validate(self.normalize(payload))
                    all_records.extend(records)
                    next_params = self.next_page(payload, current_params)
                    if not next_params:
                        break
                    current_params = next_params
                    pagination = {"has_more": True}
                except ConnectorError as exc:
                    return ConnectorResult(
                        provider=self.provider,
                        endpoint=request.endpoint,
                        status="failed",
                        records=all_records,
                        pagination={**pagination, "attempts": attempts},
                        latency_ms=latency_total,
                        error=ErrorRecord(
                            code="CONNECTOR_ERROR",
                            message=str(exc),
                            provider=self.provider,
                            retryable=True,
                        ),
                    )
                except Exception as exc:
                    return ConnectorResult(
                        provider=self.provider,
                        endpoint=request.endpoint,
                        status="failed",
                        records=all_records,
                        pagination={**pagination, "attempts": attempts},
                        latency_ms=latency_total,
                        error=ErrorRecord(
                            code="UNHANDLED_CONNECTOR_ERROR",
                            message=str(exc),
                            provider=self.provider,
                            retryable=True,
                        ),
                    )

        return ConnectorResult(
            provider=self.provider,
            endpoint=request.endpoint,
            status="ok" if all_records else "empty",
            records=all_records,
            pagination={**pagination, "attempts": attempts},
            latency_ms=latency_total,
        )

    async def _call_with_retry(
        self,
        request: APIRequest,
        params: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[httpx.Response, float]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.agent.retry_max_attempts),
            wait=wait_exponential(
                multiplier=self.settings.agent.retry_backoff_seconds,
                min=1,
                max=10,
            ),
            retry=retry_if_exception_type((httpx.HTTPError, ConnectorError)),
            reraise=True,
        ):
            with attempt:
                await self._rate_limit()
                async with httpx.AsyncClient(timeout=self.settings.llm.timeout_seconds) as client:
                    start = time.perf_counter()
                    response = await client.request(
                        request.method,
                        f"{self.base_url}{request.endpoint}",
                        params=params,
                        headers=headers,
                        json=request.json,
                    )
                    latency = round((time.perf_counter() - start) * 1000, 3)
                    if response.status_code >= 500:
                        raise ConnectorError(f"{self.provider} server error: {response.status_code}")
                    if response.status_code == 429:
                        raise ConnectorError(f"{self.provider} rate limit hit")
                    response.raise_for_status()
                    return response, latency
        raise ConnectorError("retry exhausted")
