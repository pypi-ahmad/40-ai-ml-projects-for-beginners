"""Prompt templates for planning, routing, reflection, and answer generation."""

from __future__ import annotations

from typing import Any


def planning_prompt(user_input: str, tools: list[dict[str, Any]]) -> str:
    """Prompt for decomposition and tool-aware planning."""

    return (
        "You are planning an agent task. "
        "Return strict JSON with keys: objective, steps, reasoning_summary, required_tools. "
        "Do not include markdown.\n"
        f"User request: {user_input}\n"
        f"Available tools: {tools}\n"
        "Make 1-5 concrete steps."
    )


def tool_routing_prompt(
    user_input: str,
    plan_step: str,
    tools: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> str:
    """Prompt for selecting next tool."""

    return (
        "Select best next tool. Return strict JSON with keys: tool_name, arguments, justification. "
        "If no tool needed, set tool_name to response_generator.\n"
        f"User request: {user_input}\n"
        f"Current plan step: {plan_step}\n"
        f"Prior observations: {observations}\n"
        f"Tools: {tools}"
    )


def reflection_prompt(
    user_input: str,
    plan_steps: list[str],
    observations: list[dict[str, Any]],
    errors: list[str],
) -> str:
    """Prompt for self-reflection and optional plan revision."""

    return (
        "Reflect on progress. Return strict JSON with keys: success, confidence, revised_plan, notes.\n"
        f"User request: {user_input}\n"
        f"Plan steps: {plan_steps}\n"
        f"Observations: {observations}\n"
        f"Errors: {errors}"
    )


def final_answer_prompt(
    user_input: str,
    plan_steps: list[str],
    observations: list[dict[str, Any]],
    trace_summary: list[str],
) -> str:
    """Prompt for final answer synthesis."""

    return (
        "Generate final answer grounded in observations. "
        "Return strict JSON with keys: answer, citations, completeness_score.\n"
        f"User request: {user_input}\n"
        f"Plan: {plan_steps}\n"
        f"Observations: {observations}\n"
        f"Trace summary: {trace_summary}"
    )


def parser_repair_prompt(raw_output: str, model_name: str, schema_name: str) -> str:
    """Prompt for repairing malformed JSON outputs."""

    return (
        f"Model {model_name} returned invalid JSON for schema {schema_name}. "
        "Fix to strict JSON only.\n"
        f"Invalid output:\n{raw_output}"
    )
