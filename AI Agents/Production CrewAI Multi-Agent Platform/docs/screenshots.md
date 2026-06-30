# Screenshot Checklist

Capture and store screenshots under `docs/screenshots/`:

1. `dashboard.png`
2. `crew_execution.png`
3. `agent_monitor.png`
4. `task_planner.png`
5. `workflow_graph.png`
6. `memory_viewer.png`
7. `knowledge_base.png`
8. `fastapi_swagger.png`
9. `cli_output.png`
10. `reports.png`
11. `analytics.png`

Captured in this implementation:
- `dashboard.png`
- `crew_execution.png`
- `agent_monitor.png`
- `task_planner.png`
- `workflow_graph.png`
- `memory_viewer.png`
- `knowledge_base.png`
- `fastapi_swagger.png`
- `cli_output.png`
- `reports.png`
- `analytics.png`

Capture flow command:
1. Start API: `uv run crew-platform-api`
2. Start dashboard: `uv run python app.py`
3. Run capture: `PLAYWRIGHT_BROWSERS_PATH=.playwright-browsers uv run python scripts/capture_screenshots.py`
