"""Canonical agent runtime protocol."""

from __future__ import annotations

from typing import Protocol

from rl_task_foundry.config.models import StrictModel
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.synthesis.contracts import TaskBundleContract


class SolverEpisodeInput(StrictModel):
    task_bundle: TaskBundleContract
    rendered_user_prompt: str

    @property
    def task_id(self) -> str:
        return self.task_bundle.task_id


class AgentRuntime(Protocol):
    """Common runtime contract for solver and future rollout agents."""

    async def run(self, episode: SolverEpisodeInput) -> SolverResult:
        ...
