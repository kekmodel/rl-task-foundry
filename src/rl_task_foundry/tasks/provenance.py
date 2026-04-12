"""Provenance requirement and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.tasks.models import TaskSpec


@dataclass(slots=True)
class ProvenanceCheck:
    passed: bool
    missing_requirements: list[str]


def validate_provenance(task: TaskSpec, solver_result: SolverResult) -> ProvenanceCheck:
    """Validate that the solver used the required semantic tool evidence for this task."""

    requirements = [requirement for requirement in task.provenance_requirements if requirement]
    if not requirements:
        return ProvenanceCheck(passed=True, missing_requirements=[])

    called_names, called_semantic_keys = _called_tool_evidence(solver_result.tool_trace_ref)
    called_core_tools = {name for name in called_names if name != "submit_result"}
    missing_requirements = [
        requirement
        for requirement in requirements
        if not _requirement_satisfied(
            requirement,
            called_tool_names=called_core_tools,
            called_semantic_keys=called_semantic_keys,
        )
    ]
    if not missing_requirements:
        return ProvenanceCheck(passed=True, missing_requirements=[])
    return ProvenanceCheck(
        passed=False,
        missing_requirements=missing_requirements,
    )


def _called_tool_evidence(tool_trace_ref: str) -> tuple[set[str], set[str]]:
    payload = _load_trace_payload(tool_trace_ref)
    if payload is None:
        return set(), set()
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        extracted_names = {
            str(item.get("name"))
            for item in tool_calls
            if isinstance(item, dict) and item.get("name")
        }
        extracted_semantic_keys = {
            str(item.get("semantic_key"))
            for item in tool_calls
            if isinstance(item, dict) and item.get("semantic_key")
        }
        if extracted_names or extracted_semantic_keys:
            return extracted_names, extracted_semantic_keys
    run_items = payload.get("run_items")
    if not isinstance(run_items, list):
        return set(), set()
    names: set[str] = set()
    for item in run_items:
        if not isinstance(item, str):
            continue
        match = re.search(r"tool-call\(([^)]+)\)", item)
        if match:
            names.add(match.group(1))
    return names, set()


def _requirement_satisfied(
    requirement: str,
    *,
    called_tool_names: set[str],
    called_semantic_keys: set[str],
) -> bool:
    if requirement.startswith("semantic_key:"):
        expected = requirement.removeprefix("semantic_key:")
        return expected in called_semantic_keys
    if requirement.startswith("semantic_key_prefix:"):
        expected_prefix = requirement.removeprefix("semantic_key_prefix:")
        return any(key.startswith(expected_prefix) for key in called_semantic_keys)
    return requirement in called_tool_names


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
