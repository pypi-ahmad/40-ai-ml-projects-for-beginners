# Agent Workflow

1. Request Planner selects APIs and builds execution steps.
2. API Router confirms connector mapping.
3. Authentication Agent checks required credentials per provider.
4. Data Fetch Agent executes API calls concurrently.
5. Validation Agent normalizes records and surfaces errors.
6. Reasoning Agent synthesizes findings with local Ollama model.
7. Report Generator creates Markdown/HTML/PDF/JSON/CSV outputs.
8. Memory Agent persists analysis to SQLite and semantic memory.
9. Reflection Agent decides retry vs complete.
