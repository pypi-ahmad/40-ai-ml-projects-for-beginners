"""FastAPI contract integration tests (no live client)."""

from __future__ import annotations

from multimodal_ai.api.app import create_app

EXPECTED_PATHS = {
    "/health",
    "/caption",
    "/search",
    "/retrieve",
    "/vqa",
    "/ocr",
    "/compare",
    "/analyze",
    "/documents",
    "/embeddings",
    "/analytics",
}


def test_required_routes_registered() -> None:
    app = create_app()
    route_paths = {route.path for route in app.routes}
    assert EXPECTED_PATHS.issubset(route_paths)


def test_openapi_contains_core_operations() -> None:
    app = create_app()
    schema = app.openapi()
    paths = schema.get("paths", {})
    assert "/caption" in paths
    assert "post" in paths["/caption"]
    assert "/health" in paths
    assert "get" in paths["/health"]
