"""Canonical agent runtime protocol."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.synthesis.contracts import EnvironmentContract


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SolverEpisodeInput(StrictModel):
    environment: EnvironmentContract
    instance_id: str
    rendered_user_prompt: str

    @property
    def task_id(self) -> str:
        return f"{self.environment.env_id}__{self.instance_id}"


class AgentRuntime(Protocol):
    """Common runtime contract for solver and future rollout agents."""

    async def run(self, episode: SolverEpisodeInput, *, replica_index: int) -> SolverResult:
        ...
