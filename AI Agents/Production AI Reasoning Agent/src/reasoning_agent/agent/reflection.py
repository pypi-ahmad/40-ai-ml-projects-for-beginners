"""Self-reflection and retry planning."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel

from reasoning_agent.agent.state import AgentState
from reasoning_agent.llm.ollama import OllamaProvider
from reasoning_agent.utils.output_parser import parse_structured_output
from reasoning_agent.utils.prompt_loader import load_prompt


class ReflectionDecision(BaseModel):
    """Structured reflection output."""

    strategy: str = "Proceed"
    should_retry: bool = False
    fallback: str = ""


@dataclass(slots=True)
class Reflector:
    max_retries: int = 2
    llm: OllamaProvider | None = None
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 256
    use_llm: bool = True
    retry_backoff_seconds: float = 0.0

    async def reflect(self, state: AgentState) -> AgentState:
        if state.error is None:
            state.reflection = "Execution progressing normally"
            return state

        tool_errors = [call for call in state.tool_calls if not call.success]
        retry_budget = max(self.max_retries - len(tool_errors), 0)
        decision = await self._decide(state=state, retry_budget=retry_budget)

        if decision.should_retry and retry_budget > 0:
            state.reflection = decision.strategy or (
                f"Tool failed. Retry budget remaining ({retry_budget})"
            )
            state.error = None
            state.retry_count += 1
            for step in state.plan:
                if step.status == "failed":
                    step.status = "pending"
                    break
            if self.retry_backoff_seconds > 0:
                await asyncio.sleep(self.retry_backoff_seconds)
            return state

        state.reflection = decision.strategy or "Retry budget exhausted"
        if decision.fallback:
            state.observations.append(f"fallback_note: {decision.fallback}")
        state.done = True
        return state

    async def _decide(self, state: AgentState, retry_budget: int) -> ReflectionDecision:
        if not self.use_llm or self.llm is None or self.model is None:
            return ReflectionDecision(
                strategy=f"Tool failed. Retry budget remaining ({retry_budget})",
                should_retry=retry_budget > 0,
                fallback="Provide best-effort answer from completed observations.",
            )

        prompt = load_prompt("reflection").format(
            error=state.error or "",
            observations="\n".join(state.observations[-5:]) or "None",
            retry_budget=retry_budget,
        )
        policy = load_prompt("failure_recovery").strip()
        prompt = f"{prompt}\n\n{policy}"
        try:
            response = await self.llm.generate(
                prompt=prompt,
                model=self.model,
                temperature=min(self.temperature, 0.2),
                max_tokens=self.max_tokens,
                raw=True,
            )
            decision = parse_structured_output(response.text, ReflectionDecision, retries=1)
            if retry_budget <= 0:
                decision.should_retry = False
            return decision
        except Exception:  # noqa: BLE001
            return ReflectionDecision(
                strategy=f"Tool failed. Retry budget remaining ({retry_budget})",
                should_retry=retry_budget > 0,
                fallback="Provide best-effort answer from completed observations.",
            )
