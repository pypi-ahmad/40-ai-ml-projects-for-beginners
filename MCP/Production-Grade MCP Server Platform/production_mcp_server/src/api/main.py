from __future__ import annotations

import uvicorn

from api.app import create_api_app
from server.platform import Platform


platform = Platform.from_config("configs/default.yaml")
app = create_api_app(platform)


if __name__ == "__main__":
    uvicorn.run(app, host=platform.settings.transport.host, port=platform.settings.transport.port)
