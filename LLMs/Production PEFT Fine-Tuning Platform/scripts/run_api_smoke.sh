#!/usr/bin/env bash
set -euo pipefail

uv run python - << 'PY'
from fastapi.testclient import TestClient
from peft_platform.api.app import create_app

client = TestClient(create_app())

print("health", client.get("/health").status_code)
print("models", client.get("/models").status_code)
print("generate", client.post("/generate", json={"prompt": "hello"}).status_code)
PY
