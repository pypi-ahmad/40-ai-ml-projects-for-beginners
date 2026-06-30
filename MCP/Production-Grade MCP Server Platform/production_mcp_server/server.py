from __future__ import annotations

import os
from pathlib import Path

# Make this entrypoint import-compatible with `import server.platform`.
__path__ = [str((Path(__file__).parent / "src" / "server").resolve())]


def main() -> None:
    from server.platform import Platform

    config = os.environ.get("MCP_SERVER_CONFIG", "configs/default.yaml")
    runtime = os.environ.get("MCP_SERVER_RUNTIME")
    transport = os.environ.get("MCP_SERVER_TRANSPORT")

    platform = Platform.from_config(config)
    if runtime:
        platform.settings.transport.runtime = runtime
    if transport:
        platform.settings.transport.mode = transport

    platform.run_mcp_server()


if __name__ == "__main__":
    main()
