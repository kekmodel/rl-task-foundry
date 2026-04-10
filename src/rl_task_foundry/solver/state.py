"""Canonical solver and rollout state."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AgentStep:
    observation: str
    policy_output: str
    action_type: str
    tool_name: str | None = None
    tool_input: dict[str, object] = field(default_factory=dict)
    tool_output: str | None = None
    state_update: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRuntimeState:
    task_id: str
    transcript: list[str] = field(default_factory=list)
    explicit_memory_events: list[dict[str, object]] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
