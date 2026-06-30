"""FastAPI launch entrypoint."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("api_intel_agent.api.app:app", host="0.0.0.0", port=8000, reload=False)
