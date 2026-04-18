"""Solver-side result models."""

from __future__ import annotations

from pydantic import Field

from rl_task_foundry.config.models import StrictModel


class SolverResult(StrictModel):
    task_id: str
    solver_id: str
    provider: str
    model: str
    raw_output_text: str
    structured_output: dict[str, object] | None = None
    explicit_memory_events: list[dict[str, object]] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    turn_count: int = 0
    status: str
    termination_reason: str | None = None
    termination_metadata: dict[str, object] = Field(default_factory=dict)
