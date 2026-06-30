"""LangGraph workflow wiring for reasoning agent."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from reasoning_agent.agent.state import AgentState
from reasoning_agent.executor.executor import Executor
from reasoning_agent.observability.metrics import MetricCollector
from reasoning_agent.observation.processor import ObservationProcessor
from reasoning_agent.planner.planner import Planner
from reasoning_agent.planner.reflection import Reflector
from reasoning_agent.recovery.error_handler import ErrorHandler
from reasoning_agent.response.generator import ResponseGenerator
from reasoning_agent.routing.tool_router import ToolRouter
from reasoning_agent.schemas import IterationTrace

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional runtime import
    END = "__end__"
    START = "__start__"
    StateGraph = None


@dataclass
class AgentComponents:
    """Runtime component container."""

    planner: Planner
    router: ToolRouter
    executor: Executor
    observer: ObservationProcessor
    reflector: Reflector
    responder: ResponseGenerator
    errors: ErrorHandler
    metrics: MetricCollector


class FallbackWorkflow:
    """Non-langgraph fallback loop with same node semantics."""

    def __init__(self, components: AgentComponents, tools_metadata: list[dict[str, Any]]) -> None:
        self.components = components
        self.tools_metadata = tools_metadata

    def invoke(self, state: AgentState) -> AgentState:
        state.update(planner_node(state, self.components, self.tools_metadata))
        while not state.get("done", False):
            state.update(router_node(state, self.components, self.tools_metadata))
            if state.get("selected_tool") == "response_generator":
                break
            state.update(executor_node(state, self.components))
            if state.get("last_observation_ok", True):
                state.update(observation_node(state, self.components))
                state.update(reflection_node(state, self.components))
                if state.get("done", False):
                    break
            else:
                state.update(error_node(state, self.components))
                if state.get("done", False):
                    break
        state.update(response_node(state, self.components))
        return state


def planner_node(state: AgentState, components: AgentComponents, tools_metadata: list[dict[str, Any]]) -> AgentState:
    """Create initial plan."""

    if state.get("plan_steps"):
        return {}

    started = time.perf_counter()
    plan = components.planner.plan(state["user_input"], tools_metadata)
    components.metrics.inc("planner_calls")
    components.metrics.add("planner_latency_ms", (time.perf_counter() - started) * 1000)

    return {
        "objective": plan.objective,
        "plan_steps": plan.steps,
        "thought_summary": plan.reasoning_summary,
    }


def router_node(state: AgentState, components: AgentComponents, tools_metadata: list[dict[str, Any]]) -> AgentState:
    """Select next tool/action."""

    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 8))
    if iteration >= max_iterations:
        return {"done": True, "termination_reason": "max_iterations", "selected_tool": "response_generator"}

    steps = state.get("plan_steps", [])
    current_step_idx = int(state.get("current_step", 0))
    step = steps[current_step_idx] if current_step_idx < len(steps) else state["user_input"]

    obs = [o.model_dump() for o in state.get("observations", [])]

    started = time.perf_counter()
    routed = components.router.route(
        user_input=state["user_input"],
        step=step,
        tools=tools_metadata,
        observations=obs,
    )
    components.metrics.inc("router_calls")
    components.metrics.add("router_latency_ms", (time.perf_counter() - started) * 1000)

    tool_name = routed.tool_name
    if tool_name == "response_generator":
        return {
            "selected_tool": tool_name,
            "tool_args": {},
            "done": True,
            "termination_reason": "completed",
            "thought_summary": routed.justification,
        }

    return {
        "selected_tool": tool_name,
        "tool_args": routed.arguments,
        "thought_summary": routed.justification,
    }


def executor_node(state: AgentState, components: AgentComponents) -> AgentState:
    """Execute selected tool."""

    started = time.perf_counter()
    obs = components.executor.execute(
        session_id=state["session_id"],
        run_id=state["run_id"],
        tool_name=state["selected_tool"],
        tool_args=state.get("tool_args", {}),
    )
    components.metrics.inc("tool_calls")
    components.metrics.add("tool_latency_ms", obs.latency_ms)
    components.metrics.add("executor_latency_ms", (time.perf_counter() - started) * 1000)

    observations = list(state.get("observations", []))
    observations.append(obs)

    return {
        "observations": observations,
        "last_observation_ok": obs.ok,
        "last_error": obs.error if not obs.ok else "",
    }


def observation_node(state: AgentState, components: AgentComponents) -> AgentState:
    """Persist observation and create trace iteration."""

    obs = state["observations"][-1]
    summary = components.observer.process(
        session_id=state["session_id"],
        run_id=state["run_id"],
        observation=obs,
    )

    trace = list(state.get("trace", []))
    iteration = int(state.get("iteration", 0)) + 1
    trace.append(
        IterationTrace(
            iteration=iteration,
            thought_summary=state.get("thought_summary", ""),
            action={"name": state.get("selected_tool", ""), "arguments": state.get("tool_args", {})},
            observation=obs,
            reflection="",
            retries=int(state.get("retries", 0)),
            latency_ms=float(obs.latency_ms),
        )
    )

    current_step = int(state.get("current_step", 0))
    if current_step < len(state.get("plan_steps", [])) - 1:
        current_step += 1

    return {
        "trace": trace,
        "iteration": iteration,
        "current_step": current_step,
        "observation_summary": summary,
    }


def reflection_node(state: AgentState, components: AgentComponents) -> AgentState:
    """Run reflection and optional plan revision."""

    obs = [o.model_dump() for o in state.get("observations", [])]
    errors = list(state.get("errors", []))
    result = components.reflector.reflect(
        user_input=state["user_input"],
        plan_steps=list(state.get("plan_steps", [])),
        observations=obs,
        errors=errors,
    )

    trace = list(state.get("trace", []))
    if trace:
        latest = trace[-1]
        latest.reflection = result.notes
        trace[-1] = latest

    done = bool(result.success and int(state.get("current_step", 0)) >= len(state.get("plan_steps", [])) - 1)
    termination = state.get("termination_reason", "")
    if done:
        termination = "completed"

    plan_steps = list(state.get("plan_steps", []))
    if result.revised_plan:
        plan_steps = result.revised_plan

    return {
        "trace": trace,
        "plan_steps": plan_steps,
        "done": done,
        "termination_reason": termination,
    }


def error_node(state: AgentState, components: AgentComponents) -> AgentState:
    """Retry decision for failed tool call."""

    retries = int(state.get("retries", 0))
    last_error = str(state.get("last_error", "unknown error"))
    current_step_idx = int(state.get("current_step", 0))
    steps = list(state.get("plan_steps", []))
    step = steps[current_step_idx] if current_step_idx < len(steps) else state["user_input"]

    decision = components.errors.decide(
        error=last_error,
        retries=retries,
        max_retries=int(state.get("max_retries", 2)),
        current_step=step,
    )

    errors = list(state.get("errors", []))
    errors.append(last_error)
    components.metrics.inc("tool_failures")

    if not decision.retry:
        return {
            "done": True,
            "termination_reason": "tool_failure",
            "errors": errors,
        }

    plan_steps = list(state.get("plan_steps", []))
    if decision.revised_step:
        if current_step_idx < len(plan_steps):
            plan_steps[current_step_idx] = decision.revised_step
        else:
            plan_steps.append(decision.revised_step)

    components.metrics.inc("retries")
    return {
        "retries": retries + 1,
        "plan_steps": plan_steps,
        "errors": errors,
    }


def response_node(state: AgentState, components: AgentComponents) -> AgentState:
    """Generate final answer."""

    observations = [obs.model_dump() for obs in state.get("observations", [])]
    trace_summary = [f"#{t.iteration} {t.thought_summary} | {t.reflection}" for t in state.get("trace", [])]

    started = time.perf_counter()
    final = components.responder.generate(
        user_input=state["user_input"],
        plan_steps=list(state.get("plan_steps", [])),
        observations=observations,
        trace_summary=trace_summary,
    )
    components.metrics.inc("response_calls")
    components.metrics.add("response_latency_ms", (time.perf_counter() - started) * 1000)

    if not state.get("termination_reason"):
        state["termination_reason"] = "completed"

    return {
        "answer": final.answer,
        "citations": final.citations,
        "done": True,
        "metrics": components.metrics.snapshot(),
    }


def build_workflow(components: AgentComponents, tools_metadata: list[dict[str, Any]]):
    """Build LangGraph workflow, fallback to local loop if unavailable."""

    if StateGraph is None:
        return FallbackWorkflow(components, tools_metadata)

    graph = StateGraph(AgentState)

    graph.add_node("planner", lambda state: planner_node(state, components, tools_metadata))
    graph.add_node("router", lambda state: router_node(state, components, tools_metadata))
    graph.add_node("executor", lambda state: executor_node(state, components))
    graph.add_node("observer", lambda state: observation_node(state, components))
    graph.add_node("reflector", lambda state: reflection_node(state, components))
    graph.add_node("error_handler", lambda state: error_node(state, components))
    graph.add_node("response", lambda state: response_node(state, components))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")

    def route_after_router(state: AgentState) -> str:
        return "response" if state.get("selected_tool") == "response_generator" else "executor"

    graph.add_conditional_edges("router", route_after_router, {"executor": "executor", "response": "response"})

    def route_after_executor(state: AgentState) -> str:
        return "observer" if state.get("last_observation_ok", False) else "error"

    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {"observer": "observer", "error": "error_handler"},
    )

    def route_after_reflection(state: AgentState) -> str:
        if state.get("done", False):
            return "response"
        if int(state.get("iteration", 0)) >= int(state.get("max_iterations", 8)):
            state["termination_reason"] = "max_iterations"
            return "response"
        return "router"

    graph.add_edge("observer", "reflector")
    graph.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {"response": "response", "router": "router"},
    )

    def route_after_error(state: AgentState) -> str:
        return "response" if state.get("done", False) else "router"

    graph.add_conditional_edges(
        "error_handler",
        route_after_error,
        {"response": "response", "router": "router"},
    )

    graph.add_edge("response", END)

    return graph.compile()
