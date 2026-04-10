"""Provenance requirement and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from rl_task_foundry.tasks.models import SolverResult, TaskSpec


@dataclass(slots=True)
class ProvenanceCheck:
    passed: bool
    missing_requirements: list[str]


def validate_provenance(task: TaskSpec, solver_result: SolverResult) -> ProvenanceCheck:
    """Validate that the solver used at least one required core tool for this task."""

    requirements = [requirement for requirement in task.provenance_requirements if requirement]
    if not requirements:
        return ProvenanceCheck(passed=True, missing_requirements=[])

    called_tool_names = _called_tool_names(solver_result.tool_trace_ref)
    called_core_tools = {name for name in called_tool_names if name != "submit_result"}
    if called_core_tools.intersection(requirements):
        return ProvenanceCheck(passed=True, missing_requirements=[])
    return ProvenanceCheck(
        passed=False,
        missing_requirements=list(requirements),
    )


def _called_tool_names(tool_trace_ref: str) -> set[str]:
    payload = _load_trace_payload(tool_trace_ref)
    if payload is None:
        return set()
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        extracted = {
            str(item.get("name"))
            for item in tool_calls
            if isinstance(item, dict) and item.get("name")
        }
        if extracted:
            return extracted
    run_items = payload.get("run_items")
    if not isinstance(run_items, list):
        return set()
    names: set[str] = set()
    for item in run_items:
        if not isinstance(item, str):
            continue
        match = re.search(r"tool-call\(([^)]+)\)", item)
        if match:
            names.add(match.group(1))
    return names


def _load_trace_payload(tool_trace_ref: str) -> dict[str, Any] | None:
    if not tool_trace_ref:
        return None
    path = Path(tool_trace_ref)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload
