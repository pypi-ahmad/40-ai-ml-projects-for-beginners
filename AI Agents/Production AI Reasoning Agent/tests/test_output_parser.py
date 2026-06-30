from __future__ import annotations

import pytest
from pydantic import BaseModel

from reasoning_agent.utils.output_parser import ParseError, parse_structured_output


class DemoModel(BaseModel):
    value: int


def test_parse_structured_output_valid_json() -> None:
    out = parse_structured_output('{"value": 5}', DemoModel)
    assert out.value == 5


def test_parse_structured_output_extracts_embedded_json() -> None:
    out = parse_structured_output('prefix {"value": 7} suffix', DemoModel)
    assert out.value == 7


def test_parse_structured_output_raises_on_invalid() -> None:
    with pytest.raises(ParseError):
        parse_structured_output("not json", DemoModel, retries=1)
