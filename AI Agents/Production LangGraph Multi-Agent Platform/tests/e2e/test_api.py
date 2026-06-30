from langgraph_platform.api.main import create_app


def test_api_routes_registered() -> None:
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    for path in {
        "/chat",
        "/workflow",
        "/graph",
        "/agents",
        "/tasks",
        "/memory",
        "/reports",
        "/knowledge",
        "/search",
        "/analytics",
        "/metrics",
        "/health",
    }:
        assert path in paths
