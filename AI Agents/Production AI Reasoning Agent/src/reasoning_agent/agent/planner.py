"""Planner for multi-step task decomposition."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from reasoning_agent.agent.state import AgentState, PlanStep
from reasoning_agent.llm.ollama import OllamaProvider
from reasoning_agent.utils.output_parser import parse_structured_output
from reasoning_agent.utils.prompt_loader import load_prompt


class PlanModel(BaseModel):
    """Structured planning output."""

    thoughts: list[str] = Field(default_factory=list)
    steps: list[dict[str, object]] = Field(default_factory=list)


class ToolSelectionModel(BaseModel):
    """Structured tool-selection output."""

    tool_name: str | None = None
    reason: str = ""


@dataclass(slots=True)
class Planner:
    llm: OllamaProvider
    model: str
    temperature: float
    max_tokens: int
    use_llm: bool = True

    async def build_plan(self, state: AgentState, available_tools: list[str]) -> AgentState:
        prompt_template = load_prompt("planning")
        reasoning_template = load_prompt("reasoning")
        prompt = (
            prompt_template.replace("{query}", state.query).replace(
                "{available_tools}", ", ".join(available_tools)
            )
        )
        prompt = (
            f"{prompt}\n\nReasoning mode: {state.reasoning_mode}\n"
            f"Reasoning guidance:\n{reasoning_template.strip()}"
        )

        fallback = self._heuristic_plan(state.query)
        if not self.use_llm:
            state.thoughts.append("Planner configured for heuristic mode")
            state.plan = self._filter_unavailable_tools(fallback.plan, available_tools, state)
            return state
        try:
            response = await self.llm.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                raw=True,
            )
            parsed = parse_structured_output(response.text, PlanModel, retries=2)
            steps: list[PlanStep] = []
            for idx, step in enumerate(parsed.steps, start=1):
                steps.append(
                    PlanStep(
                        step_id=str(step.get("step_id", idx)),
                        description=str(step.get("description", "Execute step")),
                        tool_name=step.get("tool_name") or None,
                        tool_input=step.get("tool_input") or {},
                        status="pending",
                    )
                )
            steps = await self._repair_plan_tools(
                state=state, steps=steps, available_tools=available_tools
            )
            if not steps:
                steps = fallback.plan
            state.plan = steps
            state.thoughts.extend(parsed.thoughts or ["Plan generated with model"])
            return state
        except Exception:  # noqa: BLE001
            state.thoughts.append("Planner fallback engaged")
            state.plan = self._filter_unavailable_tools(fallback.plan, available_tools, state)
            return state

    def _heuristic_plan(self, query: str) -> AgentState:
        state = AgentState(query=query)
        lower = query.lower()
        if "convert" in lower and any(token in lower for token in ["km", "kilometer", "mile", "miles"]):
            value = self._extract_first_number(query) or 1.0
            from_unit = "km" if "km" in lower or "kilometer" in lower else "m"
            to_unit = "mi" if "mile" in lower else "km"
            state.plan = [
                PlanStep(
                    step_id="1",
                    description="Convert units",
                    tool_name="unit_converter",
                    tool_input={"value": value, "from_unit": from_unit, "to_unit": to_unit},
                )
            ]
            return state

        if "convert" in lower and any(token in lower for token in ["usd", "eur", "inr", "gbp", "jpy"]):
            value = self._extract_first_number(query) or 1.0
            from_currency = self._extract_currency(lower, fallback="USD")
            to_currency = self._extract_currency(lower, fallback="INR", pick_last=True)
            state.plan = [
                PlanStep(
                    step_id="1",
                    description="Convert currency",
                    tool_name="currency_converter",
                    tool_input={
                        "amount": value,
                        "from_currency": from_currency,
                        "to_currency": to_currency,
                    },
                )
            ]
            return state

        if any(token in lower for token in ["calculate", "math", "+", "-", "*", "/"]):
            expression = self._extract_math_expression(query)
            state.plan = [
                PlanStep(
                    step_id="1",
                    description="Solve math expression",
                    tool_name="calculator",
                    tool_input={"expression": expression},
                )
            ]
            return state
        if any(token in lower for token in ["weather", "temperature", "forecast"]):
            state.plan = [
                PlanStep(
                    step_id="1",
                    description="Get weather summary",
                    tool_name="weather",
                    tool_input={"location": query},
                )
            ]
            return state
        if any(token in lower for token in ["search", "who", "what", "when", "where", "news"]):
            state.plan = [
                PlanStep(
                    step_id="1",
                    description="Search web sources",
                    tool_name="duckduckgo_search",
                    tool_input={"query": query, "max_results": 5},
                )
            ]
            return state

        state.plan = [
            PlanStep(
                step_id="1",
                description="No external tool needed. Answer directly.",
                tool_name=None,
                tool_input={},
            )
        ]
        return state

    def _extract_math_expression(self, query: str) -> str:
        cleaned = re.sub(r"(?i)\bcalculate\b", "", query).strip()
        match = re.search(r"[-+/*().\d\s]+", cleaned)
        if match and any(ch.isdigit() for ch in match.group(0)):
            return match.group(0).strip()
        fallback = re.sub(r"[^0-9+\-*/(). ]", "", cleaned).strip()
        return fallback or cleaned

    def _extract_first_number(self, text: str) -> float | None:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        return float(match.group(0))

    def _extract_currency(self, text: str, fallback: str, pick_last: bool = False) -> str:
        currencies = re.findall(r"\b(?:usd|eur|inr|gbp|jpy)\b", text)
        if not currencies:
            return fallback
        return currencies[-1].upper() if pick_last else currencies[0].upper()

    def serialize_plan(self, state: AgentState) -> str:
        return json.dumps([step.model_dump(mode="json") for step in state.plan], ensure_ascii=False)

    async def _repair_plan_tools(
        self,
        state: AgentState,
        steps: list[PlanStep],
        available_tools: list[str],
    ) -> list[PlanStep]:
        """Normalize invalid tool references and attempt LLM tool repair for ambiguous steps."""

        if not steps:
            return steps
        available_set = set(available_tools)
        tool_prompt = load_prompt("tool_selection")

        for step in steps:
            if step.tool_name and step.tool_name not in available_set:
                state.thoughts.append(
                    f"Planner proposed unavailable tool '{step.tool_name}', clearing assignment"
                )
                step.tool_name = None
            if step.tool_name is not None or not available_tools:
                continue
            if not self._step_likely_needs_tool(state.query, step.description):
                continue

            tool_selection_prompt = (
                tool_prompt.replace("{query}", state.query)
                .replace("{step_description}", step.description)
                .replace("{available_tools}", ", ".join(available_tools))
            )
            try:
                response = await self.llm.generate(
                    prompt=tool_selection_prompt,
                    model=self.model,
                    temperature=min(self.temperature, 0.2),
                    max_tokens=256,
                    raw=True,
                )
                selected = parse_structured_output(
                    response.text, ToolSelectionModel, retries=1
                ).tool_name
                if selected in available_set:
                    step.tool_name = selected
                    state.thoughts.append(f"Tool selector assigned '{selected}' to step {step.step_id}")
            except Exception:  # noqa: BLE001
                continue
        return steps

    def _step_likely_needs_tool(self, query: str, description: str) -> bool:
        combined = f"{query} {description}".lower()
        indicators = (
            "calculate",
            "search",
            "weather",
            "convert",
            "file",
            "csv",
            "json",
            "read",
            "web",
            "wikipedia",
            "today",
            "latest",
        )
        return any(token in combined for token in indicators)

    def _filter_unavailable_tools(
        self, steps: list[PlanStep], available_tools: list[str], state: AgentState
    ) -> list[PlanStep]:
        available = set(available_tools)
        for step in steps:
            if step.tool_name and step.tool_name not in available:
                state.thoughts.append(
                    f"Heuristic step requested unavailable tool '{step.tool_name}', switching to no-tool step"
                )
                step.tool_name = None
                step.tool_input = {}
        return steps
