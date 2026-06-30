from __future__ import annotations

from functools import lru_cache

from server.platform import Platform


@lru_cache(maxsize=1)
def get_platform(config_path: str = "configs/default.yaml") -> Platform:
    return Platform.from_config(config_path)
