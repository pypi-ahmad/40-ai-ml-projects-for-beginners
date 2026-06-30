from __future__ import annotations

from reasoning_agent.parsing.output_parser import StructuredOutputParser
from reasoning_agent.recovery.error_handler import ErrorHandler
from reasoning_agent.routing.tool_router import ToolRouter
from reasoning_agent.schemas import PlanningOutput


class _FakeLLM:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs

    def generate(self, model: str, prompt: str, **kwargs):
        class R:
            def __init__(self, text: str):
                self.text = text
                self.latency_ms = 1.0
                self.model = model

        if not self.outputs:
            return R('{"objective":"x","steps":["a"],"reasoning_summary":"ok","required_tools":[]}')
        return R(self.outputs.pop(0))


def test_parser_repair_flow() -> None:
    llm = _FakeLLM(['{"objective":"o","steps":["s"],"reasoning_summary":"r","required_tools":[]}'])
    parser = StructuredOutputParser(llm, model="m", retries=1)
    out = parser.parse("not-json", PlanningOutput)
    assert out.objective == "o"


def test_tool_router_heuristics() -> None:
    llm = _FakeLLM(["invalid"])
    router = ToolRouter(llm, model="m", temperature=0.0, max_tokens=10)
    routed = router.route(user_input="Calculate 2+2", step="math", tools=[], observations=[])
    assert routed.tool_name == "calculator"


def test_error_handler_retry_and_stop() -> None:
    h = ErrorHandler()
    d1 = h.decide(error="validation failed", retries=0, max_retries=2, current_step="step")
    assert d1.retry is True
    d2 = h.decide(error="timeout", retries=2, max_retries=2, current_step="step")
    assert d2.retry is False
