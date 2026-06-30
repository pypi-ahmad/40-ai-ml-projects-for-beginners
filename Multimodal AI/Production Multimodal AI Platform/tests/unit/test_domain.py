"""Domain contract tests."""

from __future__ import annotations

from multimodal_ai.domain import RequestEnvelope, ResponseEnvelope


def test_request_envelope_defaults() -> None:
    payload = RequestEnvelope()
    assert payload.trace.trace_id
    assert payload.options == {}


def test_response_envelope_structure() -> None:
    response = ResponseEnvelope(status="ok", trace_id="t1")
    assert response.status == "ok"
    assert response.trace_id == "t1"
