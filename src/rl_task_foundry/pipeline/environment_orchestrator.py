"""Environment-contract-first rollout orchestration.

This module intentionally avoids imports from the legacy task/tool/truth stack.
It is the authoritative rollout path for synthesized environments and their
materialized instances.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from rl_task_foundry.calibration.banding import PassRateBand, clopper_pearson_interval
from rl_task_foundry.calibration.runner import calibration_decision
from rl_task_foundry.config.models import AppConfig, ProviderConfig, SolverModelConfig
from rl_task_foundry.infra.db import DatabasePools
from rl_task_foundry.solver.backend_openai_agents import OpenAIAgentsSolverBackend
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.solver.runtime import AgentRuntime, SolverEpisodeInput
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle
from rl_task_foundry.synthesis.canonicalize import RewardResult, compute_reward
from rl_task_foundry.synthesis.contracts import EnvironmentContract
from rl_task_foundry.synthesis.runtime import (
    MaterializedCanonicalAnswerRecord,
    MaterializedInstanceRecord,
    SynthesisEnvironmentDraft,
)

ToolExecutor = Callable[[dict[str, Any]], Any]
EnvironmentRuntimeFactory = Callable[
    [
        SolverModelConfig,
        ProviderConfig,
        EnvironmentContract,
        list[dict[str, Any]],
        dict[str, ToolExecutor],
    ],
    AgentRuntime,
]
EnvironmentToolExecutorFactory = Callable[
    [AtomicToolBundle],
    dict[str, ToolExecutor] | Awaitable[dict[str, ToolExecutor]],
]


@dataclass(frozen=True, slots=True)
class EnvironmentRolloutBundle:
    environment: EnvironmentContract
    atomic_tool_bundle: AtomicToolBundle
    instances: tuple[MaterializedInstanceRecord, ...]
    canonical_answers: tuple[MaterializedCanonicalAnswerRecord, ...]

    @classmethod
    def from_draft(cls, draft: SynthesisEnvironmentDraft) -> "EnvironmentRolloutBundle":
        return cls(
            environment=draft.environment,
            atomic_tool_bundle=draft.atomic_tool_bundle,
            instances=tuple(draft.instances),
            canonical_answers=tuple(draft.canonical_answers),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentSolverRun:
    env_id: str
    instance_id: str
    solver_id: str
    replica_index: int
    solver_result: SolverResult
    reward_result: RewardResult


@dataclass(frozen=True, slots=True)
class EnvironmentRolloutSummary:
    env_id: str
    db_id: str
    total_instances: int
    total_solver_runs: int
    matched_solver_runs: int
    runs: tuple[EnvironmentSolverRun, ...] = ()

    @property
    def pass_rate(self) -> float:
        if self.total_solver_runs == 0:
            return 0.0
        return self.matched_solver_runs / self.total_solver_runs


class EnvironmentQualityGateStatus(StrEnum):
    ACCEPT = "accept"
    REJECT_TOO_HARD = "reject_too_hard"
    REJECT_TOO_EASY = "reject_too_easy"


@dataclass(frozen=True, slots=True)
class EnvironmentQualityGateSummary:
    status: EnvironmentQualityGateStatus
    pass_rate: float
    matched_solver_runs: int
    total_solver_runs: int
    ci_lower: float
    ci_upper: float
    band_lower: float
    band_upper: float


@dataclass(slots=True)
class EnvironmentOrchestrator:
    config: AppConfig
    runtime_factory: EnvironmentRuntimeFactory | None = None
    tool_executor_factory: EnvironmentToolExecutorFactory | None = None
    _provider_semaphores: dict[str, asyncio.Semaphore] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _tool_executor_cache: dict[str, dict[str, ToolExecutor]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _database_pools: DatabasePools | None = field(default=None, init=False, repr=False)
    _atomic_tool_materializer: AtomicToolMaterializer | None = field(
        default=None,
        init=False,
        repr=False,
    )

    async def run_draft(self, draft: SynthesisEnvironmentDraft) -> EnvironmentRolloutSummary:
        return await self.run_bundle(EnvironmentRolloutBundle.from_draft(draft))

    async def run_bundle(self, bundle: EnvironmentRolloutBundle) -> EnvironmentRolloutSummary:
        canonical_by_instance = _canonical_answers_by_instance(bundle)
        tool_definitions = bundle.atomic_tool_bundle.actor_tool_definitions()
        tool_executors = await self._tool_executors(bundle.atomic_tool_bundle)
        runs: list[EnvironmentSolverRun] = []
        calls = []
        for instance in bundle.instances:
            canonical_record = canonical_by_instance.get(instance.instance_id)
            if canonical_record is None:
                raise ValueError(
                    f"missing canonical answer for instance_id={instance.instance_id!r}"
                )
            for solver_config in self.config.models.solvers:
                provider_config = self.config.providers[solver_config.provider]
                for replica_index in range(solver_config.replicas):
                    runtime = self._runtime_for_solver(
                        solver_config,
                        provider_config,
                        bundle.environment,
                        tool_definitions,
                        tool_executors,
                    )
                    calls.append(
                        self._run_solver(
                            runtime=runtime,
                            solver_config=solver_config,
                            instance=instance,
                            environment=bundle.environment,
                            canonical_record=canonical_record,
                            replica_index=replica_index,
                        )
                    )
        if calls:
            runs = list(await asyncio.gather(*calls))
        matched_solver_runs = sum(
            1 for run in runs if run.reward_result.status == "matched"
        )
        return EnvironmentRolloutSummary(
            env_id=bundle.environment.env_id,
            db_id=bundle.environment.db_id,
            total_instances=len(bundle.instances),
            total_solver_runs=len(runs),
            matched_solver_runs=matched_solver_runs,
            runs=tuple(runs),
        )

    async def close(self) -> None:
        if self._database_pools is not None:
            await self._database_pools.close()
            self._database_pools = None

    async def _run_solver(
        self,
        *,
        runtime: AgentRuntime,
        solver_config: SolverModelConfig,
        environment: EnvironmentContract,
        instance: MaterializedInstanceRecord,
        canonical_record: MaterializedCanonicalAnswerRecord,
        replica_index: int,
    ) -> EnvironmentSolverRun:
        episode = SolverEpisodeInput(
            environment=environment,
            instance_id=instance.instance_id,
            rendered_user_prompt=instance.rendered_user_prompt,
        )
        provider_semaphore = self._provider_semaphore(solver_config.provider)
        async with provider_semaphore:
            solver_result = await runtime.run(episode, replica_index=replica_index)
        reward_result = compute_reward(
            submitted_answer_text=solver_result.raw_output_text,
            canonical_answer=canonical_record.canonical_answer,
            output_schema=environment.task.output_schema,
        )
        return EnvironmentSolverRun(
            env_id=environment.env_id,
            instance_id=instance.instance_id,
            solver_id=solver_config.solver_id,
            replica_index=replica_index,
            solver_result=solver_result,
            reward_result=reward_result,
        )

    def _provider_semaphore(self, provider_name: str) -> asyncio.Semaphore:
        semaphore = self._provider_semaphores.get(provider_name)
        if semaphore is None:
            provider_config = self.config.providers[provider_name]
            semaphore = asyncio.Semaphore(provider_config.max_concurrency)
            self._provider_semaphores[provider_name] = semaphore
        return semaphore

    def _runtime_for_solver(
        self,
        solver_config: SolverModelConfig,
        provider_config: ProviderConfig,
        environment: EnvironmentContract,
        tool_definitions: list[dict[str, Any]],
        tool_executors: dict[str, ToolExecutor],
    ) -> AgentRuntime:
        if self.runtime_factory is not None:
            return self.runtime_factory(
                solver_config,
                provider_config,
                environment,
                tool_definitions,
                tool_executors,
            )
        if solver_config.backend != "openai_agents":
            raise NotImplementedError(f"Unsupported solver backend: {solver_config.backend}")
        runtime_config = self.config.solver_runtime.model_copy(
            update={"max_turns": environment.rollout_constraints.max_turns}
        )
        return OpenAIAgentsSolverBackend(
            solver_config=solver_config,
            provider_config=provider_config,
            runtime_config=runtime_config,
            tool_definitions=tool_definitions,
            tool_executors=tool_executors,
            session_db_path=self.config.output.traces_dir / "sessions.sqlite",
            traces_dir=self.config.output.traces_dir,
        )

    async def _tool_executors(
        self,
        bundle: AtomicToolBundle,
    ) -> dict[str, ToolExecutor]:
        cached = self._tool_executor_cache.get(bundle.db_id)
        if cached is not None:
            return cached
        if self.tool_executor_factory is not None:
            executors = self.tool_executor_factory(bundle)
            if inspect.isawaitable(executors):
                executors = await executors
            resolved = dict(executors)
            self._tool_executor_cache[bundle.db_id] = resolved
            return resolved

        pools = await self._database_pools_for_tools()
        materializer = self._tool_materializer()
        materialization = materializer.materialize_bundle(bundle)
        module = _load_atomic_tool_module(
            materialization.source_path,
            module_name=f"rl_task_foundry_atomic_tools_{bundle.db_id}",
        )
        resolved = {
            tool.name: _bind_atomic_tool_executor(
                module=module,
                tool_name=tool.name,
                pools=pools,
            )
            for tool in bundle.tools
        }
        self._tool_executor_cache[bundle.db_id] = resolved
        return resolved

    async def _database_pools_for_tools(self) -> DatabasePools:
        if self._database_pools is None:
            self._database_pools = await DatabasePools.create(self.config.database)
        return self._database_pools

    def _tool_materializer(self) -> AtomicToolMaterializer:
        if self._atomic_tool_materializer is None:
            self._atomic_tool_materializer = AtomicToolMaterializer.for_config(self.config)
        return self._atomic_tool_materializer


def _canonical_answers_by_instance(
    bundle: EnvironmentRolloutBundle,
) -> dict[str, MaterializedCanonicalAnswerRecord]:
    canonical_by_instance = {
        record.instance_id: record
        for record in bundle.canonical_answers
    }
    if len(canonical_by_instance) != len(bundle.canonical_answers):
        raise ValueError("canonical answer records must not reuse instance_id values")
    return canonical_by_instance


def evaluate_rollout_summary(
    config: AppConfig,
    summary: EnvironmentRolloutSummary,
) -> EnvironmentQualityGateSummary:
    if summary.total_solver_runs <= 0:
        raise ValueError("environment rollout summary must include at least one solver run")

    band = PassRateBand(
        lower=config.calibration.lower_pass_rate,
        upper=config.calibration.upper_pass_rate,
    )
    interval = clopper_pearson_interval(
        successes=summary.matched_solver_runs,
        trials=summary.total_solver_runs,
        alpha=config.calibration.ci_alpha,
    )
    if config.calibration.safe_early_termination:
        decision = calibration_decision(
            total_replicas=summary.total_solver_runs,
            results=[run.reward_result.status == "matched" for run in summary.runs],
            band=band,
            ci_alpha=config.calibration.ci_alpha,
        )
    else:
        decision = "continue"

    if decision == "continue":
        if band.contains(summary.pass_rate):
            status = EnvironmentQualityGateStatus.ACCEPT
        elif summary.pass_rate < band.lower:
            status = EnvironmentQualityGateStatus.REJECT_TOO_HARD
        else:
            status = EnvironmentQualityGateStatus.REJECT_TOO_EASY
    elif decision == "accept":
        status = EnvironmentQualityGateStatus.ACCEPT
    elif decision == "reject_too_hard":
        status = EnvironmentQualityGateStatus.REJECT_TOO_HARD
    elif decision == "reject_too_easy":
        status = EnvironmentQualityGateStatus.REJECT_TOO_EASY
    else:  # pragma: no cover - defensive against future calibration enum changes
        raise ValueError(f"unsupported calibration decision: {decision}")

    return EnvironmentQualityGateSummary(
        status=status,
        pass_rate=summary.pass_rate,
        matched_solver_runs=summary.matched_solver_runs,
        total_solver_runs=summary.total_solver_runs,
        ci_lower=interval.lower,
        ci_upper=interval.upper,
        band_lower=band.lower,
        band_upper=band.upper,
    )


def _load_atomic_tool_module(source_path: Path, *, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load atomic tool module: {source_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_atomic_tool_executor(
    *,
    module: ModuleType,
    tool_name: str,
    pools: DatabasePools,
) -> ToolExecutor:
    function = getattr(module, tool_name)

    async def _execute(kwargs: dict[str, Any]) -> Any:
        async with pools.solver_connection() as conn:
            result = function(conn, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

    return _execute
