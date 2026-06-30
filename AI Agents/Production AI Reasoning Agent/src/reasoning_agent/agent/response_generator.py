"""Final answer generation from trajectory."""

from __future__ import annotations

from dataclasses import dataclass

from reasoning_agent.agent.state import AgentState
from reasoning_agent.llm.ollama import OllamaProvider
from reasoning_agent.utils.prompt_loader import load_prompt


@dataclass(slots=True)
class ResponseGenerator:
    llm: OllamaProvider
    model: str
    temperature: float
    max_tokens: int
    use_llm: bool = True

    async def generate(self, state: AgentState) -> AgentState:
        if not self.use_llm:
            state.final_answer = self._fallback_answer(state)
            return state
        prompt = load_prompt("final_answer").format(
            query=state.query,
            observations="\n".join(state.observations[-8:]),
            reflection=state.reflection,
        )
        try:
            response = await self.llm.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                raw=True,
            )
            state.final_answer = response.text
        except Exception:  # noqa: BLE001
            state.final_answer = self._fallback_answer(state)
        return state

    def _fallback_answer(self, state: AgentState) -> str:
        if state.observations:
            return f"Based on available observations: {state.observations[-1]}"
        if state.error:
            return f"I could not complete request due to: {state.error}"
        return "I could not gather enough information."
