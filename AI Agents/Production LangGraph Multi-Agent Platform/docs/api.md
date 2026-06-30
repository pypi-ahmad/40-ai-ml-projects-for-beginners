# API Reference

## Endpoints

- `POST /chat` - run workflow in chat mode
- `POST /workflow` - explicit workflow execution
- `GET /graph` - graph topology and branch metadata
- `GET /agents` - enterprise agent registry
- `POST /tasks` - HITL actions (approve/reject/pause/resume/rerun/override)
- `GET /memory` - inspect persistent memory
- `POST /reports` - export report to markdown/html/pdf/json
- `POST /knowledge` - ingest files/URLs into RAG knowledge base
- `POST /search` - direct web search tool endpoint
- `GET /analytics` - aggregate execution analytics
- `GET /metrics` - runtime CPU/GPU metrics
- `GET /health` - service health check

## MCP Endpoints

- `GET /mcp/capabilities`
- `POST /mcp/call`
