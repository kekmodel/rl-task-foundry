"""Canonical agent runtime protocol."""

from __future__ import annotations

from typing import Protocol

from rl_task_foundry.tasks.models import SolverResult, TaskSpec


class AgentRuntime(Protocol):
    """Common runtime contract for solver and future rollout agents."""

    async def run(self, task: TaskSpec, *, replica_index: int) -> SolverResult:
        ...
