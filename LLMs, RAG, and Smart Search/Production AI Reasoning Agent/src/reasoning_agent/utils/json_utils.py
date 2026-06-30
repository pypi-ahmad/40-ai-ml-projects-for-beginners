"""JSON helpers with deterministic encoding."""

from __future__ import annotations

from typing import Any

import orjson


def dumps(data: Any, *, indent: bool = False) -> str:
    """Serialize JSON with stable options."""

    option = orjson.OPT_SORT_KEYS
    if indent:
        option |= orjson.OPT_INDENT_2
    return orjson.dumps(data, option=option).decode("utf-8")


def loads(payload: str) -> Any:
    """Deserialize JSON safely."""

    return orjson.loads(payload)
