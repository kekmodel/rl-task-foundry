"""Task-bundle-first rollout orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import Any

from rl_task_foundry.calibration.banding import PassRateBand, clopper_pearson_interval
from rl_task_foundry.calibration.runner import calibration_decision
from rl_task_foundry.config.models import AppConfig, ProviderConfig, SolverModelConfig
from rl_task_foundry.infra.db import DatabasePools, ensure_attached_database_pools
from rl_task_foundry.solver.backend_openai_agents import OpenAIAgentsSolverBackend
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.solver.runtime import AgentRuntime, SolverEpisodeInput
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle
from rl_task_foundry.synthesis.canonicalize import RewardResult, canonical_json, compute_reward
from rl_task_foundry.synthesis.contracts import TaskBundleContract
from rl_task_foundry.synthesis.runtime import SynthesisTaskDraft
from rl_task_foundry.synthesis.tool_runtime import (
    ToolExecutor,
    bind_atomic_tool_executor,
    build_shuffle_seed,
    load_atomic_tool_module,
    with_tool_shuffle_seed,
)

TaskRuntimeFactory = Callable[
    [
        SolverModelConfig,
        ProviderConfig,
        TaskBundleContract,
        list[dict[str, Any]],
        dict[str, ToolExecutor],
    ],
    AgentRuntime,
]
TaskToolExecutorFactory = Callable[
    [AtomicToolBundle],
    dict[str, ToolExecutor] | Awaitable[dict[str, ToolExecutor]],
]
TaskSolverRunFactory = Callable[[], Awaitable["TaskSolverRun"]]


@dataclass(frozen=True, slots=True)
class TaskRolloutBundle:
    task_bundle: TaskBundleContract
    atomic_tool_bundle: AtomicToolBundle
    rendered_user_prompt: str
    canonical_answer_json: str
    label_signature: str

    @classmethod
    def from_draft(cls, draft: SynthesisTaskDraft) -> "TaskRolloutBundle":
        return cls(
            task_bundle=draft.task_bundle,
            atomic_tool_bundle=draft.atomic_tool_bundle,
            rendered_user_prompt=draft.rendered_user_prompt,
            canonical_answer_json=draft.canonical_answer_json,
            label_signature=draft.label_signature,
        )


@dataclass(frozen=True, slots=True)
class TaskSolverRun:
    task_id: str
    solver_id: str
    solver_index: int
    solver_result: SolverResult
    reward_result: RewardResult


@dataclass(frozen=True, slots=True)
class TaskRolloutSummary:
    task_id: str
    db_id: str
    planned_solver_runs: int
    total_solver_runs: int
    matched_solver_runs: int
    early_stop_decision: str | None = None
    runs: tuple[TaskSolverRun, ...] = ()

    @property
    def pass_rate(self) -> float:
        if self.total_solver_runs == 0:
            return 0.0
        return self.matched_solver_runs / self.total_solver_runs


class TaskQualityGateStatus(StrEnum):
    ACCEPT = "accept"
    REJECT_TOO_HARD = "reject_too_hard"
    REJECT_TOO_EASY = "reject_too_easy"


@dataclass(frozen=True, slots=True)
class TaskQualityGateSummary:
    status: TaskQualityGateStatus
    pass_rate: float
    matched_solver_runs: int
    total_solver_runs: int
    ci_lower: float
    ci_upper: float
    band_lower: float
    band_upper: float
    unique_answers: int = 0
    divergence_ratio: float = 0.0


@dataclass(slots=True)
class SolverOrchestrator:
    config: AppConfig
    runtime_factory: TaskRuntimeFactory | None = None
    tool_executor_factory: TaskToolExecutorFactory | None = None
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

    async def run_draft(self, draft: SynthesisTaskDraft) -> TaskRolloutSummary:
        return await self.run_bundle(TaskRolloutBundle.from_draft(draft))

    async def run_bundle(self, bundle: TaskRolloutBundle) -> TaskRolloutSummary:
        tool_definitions = bundle.atomic_tool_bundle.actor_tool_definitions()
        tool_executors = await self._tool_executors(bundle.atomic_tool_bundle)
        runs: list[TaskSolverRun] = []
        calls: list[TaskSolverRunFactory] = []
        for solver_index, solver_config in enumerate(self.config.models.solvers):
            provider_config = self.config.providers[solver_config.provider]
            shuffle_seed = build_shuffle_seed(
                "solver",
                bundle.task_bundle.task_id,
                solver_config.solver_id,
                solver_index,
            )
            seeded_tool_executors = {
                name: with_tool_shuffle_seed(executor, shuffle_seed=shuffle_seed)
                for name, executor in tool_executors.items()
            }
            runtime = self._runtime_for_solver(
                solver_config,
                provider_config,
                bundle.task_bundle,
                tool_definitions,
                seeded_tool_executors,
            )
            calls.append(
                partial(
                    self._run_solver,
                    runtime=runtime,
                    solver_config=solver_config,
                    solver_index=solver_index,
                    task_bundle=bundle.task_bundle,
                    rendered_user_prompt=bundle.rendered_user_prompt,
                    canonical_answer_json=bundle.canonical_answer_json,
                )
            )

        planned_solver_runs = min(len(calls), self.config.calibration.max_solver_runs)
        scheduled_calls = calls[:planned_solver_runs]
        early_stop_decision: str | None = None
        if scheduled_calls:
            runs, early_stop_decision = await self._execute_solver_batches(scheduled_calls)
        matched_solver_runs = sum(1 for run in runs if run.reward_result.status == "matched")
        return TaskRolloutSummary(
            task_id=bundle.task_bundle.task_id,
            db_id=bundle.task_bundle.db_id,
            planned_solver_runs=planned_solver_runs,
            total_solver_runs=len(runs),
            matched_solver_runs=matched_solver_runs,
            early_stop_decision=early_stop_decision,
            runs=tuple(runs),
        )

    async def close(self) -> None:
        if self._database_pools is not None:
            await self._database_pools.close()
            self._database_pools = None
        self._tool_executor_cache.clear()
        OpenAIAgentsSolverBackend.clear_model_cache()

    async def _run_solver(
        self,
        *,
        runtime: AgentRuntime,
        solver_config: SolverModelConfig,
        solver_index: int,
        task_bundle: TaskBundleContract,
        rendered_user_prompt: str,
        canonical_answer_json: str,
    ) -> TaskSolverRun:
        episode = SolverEpisodeInput(
            task_bundle=task_bundle,
            rendered_user_prompt=rendered_user_prompt,
        )
        provider_semaphore = self._provider_semaphore(solver_config.provider)
        async with provider_semaphore:
            try:
                solver_result = await runtime.run(episode)
            except Exception as exc:
                solver_result = SolverResult(
                    task_id=episode.task_id,
                    solver_id=solver_config.solver_id,
                    provider=solver_config.provider,
                    model=solver_config.model,
                    transcript_ref="memory://solver-error/transcript",
                    tool_trace_ref="memory://solver-error/tools",
                    raw_output_text="",
                    structured_output=None,
                    explicit_memory_events=[],
                    token_usage={},
                    latency_ms=0,
                    turn_count=0,
                    status="failed",
                    termination_reason=exc.__class__.__name__,
                    termination_metadata={"detail": str(exc)},
                )
        reward_result = compute_reward(
            submitted_answer_text=solver_result.raw_output_text,
            canonical_answer=json.loads(canonical_answer_json),
            output_schema=task_bundle.task.output_schema,
        )
        return TaskSolverRun(
            task_id=task_bundle.task_id,
            solver_id=solver_config.solver_id,
            solver_index=solver_index,
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
        task_bundle: TaskBundleContract,
        tool_definitions: list[dict[str, Any]],
        tool_executors: dict[str, ToolExecutor],
    ) -> AgentRuntime:
        if self.runtime_factory is not None:
            return self.runtime_factory(
                solver_config,
                provider_config,
                task_bundle,
                tool_definitions,
                tool_executors,
            )
        if solver_config.backend != "openai_agents":
            raise NotImplementedError(f"Unsupported solver backend: {solver_config.backend}")
        runtime_config = self.config.solver_runtime.model_copy(
            update={"max_turns": task_bundle.rollout_constraints.max_turns}
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
        module = load_atomic_tool_module(
            materialization.source_path,
            module_name=f"rl_task_foundry_atomic_tools_{bundle.db_id}",
        )
        resolved = {
            tool.name: bind_atomic_tool_executor(
                module=module,
                tool_name=tool.name,
                pools=pools,
            )
            for tool in bundle.tools
        }
        self._tool_executor_cache[bundle.db_id] = resolved
        return resolved

    async def _execute_solver_batches(
        self,
        calls: list[TaskSolverRunFactory],
    ) -> tuple[list[TaskSolverRun], str | None]:
        runs: list[TaskSolverRun] = []
        total_solver_runs = len(calls)
        if total_solver_runs == 0:
            return runs, None

        batch_size = max(
            1,
            min(self.config.calibration.solver_batch_size, total_solver_runs),
        )
        cursor = 0
        early_stop_decision: str | None = None
        band = PassRateBand(
            lower=self.config.calibration.lower_pass_rate,
            upper=self.config.calibration.upper_pass_rate,
        )

        while cursor < total_solver_runs:
            batch = calls[cursor : cursor + batch_size]
            cursor += len(batch)
            runs.extend(await asyncio.gather(*(call() for call in batch)))
            if not self.config.calibration.safe_early_termination:
                continue
            early_stop_decision = calibration_decision(
                total_solver_runs=total_solver_runs,
                results=[run.reward_result.status == "matched" for run in runs],
                band=band,
                ci_alpha=self.config.calibration.ci_alpha,
            )
            if early_stop_decision != "continue":
                break

        if early_stop_decision == "continue":
            early_stop_decision = None
        return runs, early_stop_decision

    async def _database_pools_for_tools(self) -> DatabasePools:
        return await ensure_attached_database_pools(
            self,
            attr_name="_database_pools",
            config=self.config.database,
        )

    def _tool_materializer(self) -> AtomicToolMaterializer:
        if self._atomic_tool_materializer is None:
            self._atomic_tool_materializer = AtomicToolMaterializer.for_config(self.config)
        return self._atomic_tool_materializer


def _solver_divergence(summary: TaskRolloutSummary) -> tuple[int, float]:
    answers: set[str] = set()
    submitted = 0
    for run in summary.runs:
        text = run.solver_result.raw_output_text
        if text:
            answers.add(canonical_json(json.loads(text)) if text.strip().startswith(("{", "[")) else text)
            submitted += 1
    unique = len(answers)
    ratio = unique / submitted if submitted > 0 else 0.0
    return unique, ratio


def evaluate_rollout_summary(
    config: AppConfig,
    summary: TaskRolloutSummary,
) -> TaskQualityGateSummary:
    if summary.total_solver_runs <= 0:
        raise ValueError("task rollout summary must include at least one solver run")

    band = PassRateBand(
        lower=config.calibration.lower_pass_rate,
        upper=config.calibration.upper_pass_rate,
    )
    interval = clopper_pearson_interval(
        successes=summary.matched_solver_runs,
        trials=summary.total_solver_runs,
        alpha=config.calibration.ci_alpha,
    )
    if summary.early_stop_decision is not None:
        decision = summary.early_stop_decision
    elif config.calibration.safe_early_termination:
        decision = calibration_decision(
            total_solver_runs=summary.planned_solver_runs,
            results=[run.reward_result.status == "matched" for run in summary.runs],
            band=band,
            ci_alpha=config.calibration.ci_alpha,
        )
    else:
        decision = "continue"

    if decision == "continue":
        if band.contains(summary.pass_rate):
            status = TaskQualityGateStatus.ACCEPT
        elif summary.pass_rate < band.lower:
            status = TaskQualityGateStatus.REJECT_TOO_HARD
        else:
            status = TaskQualityGateStatus.REJECT_TOO_EASY
    elif decision == "accept":
        status = TaskQualityGateStatus.ACCEPT
    elif decision == "reject_too_hard":
        status = TaskQualityGateStatus.REJECT_TOO_HARD
    elif decision == "reject_too_easy":
        status = TaskQualityGateStatus.REJECT_TOO_EASY
    else:  # pragma: no cover
        raise ValueError(f"unsupported calibration decision: {decision}")

    unique_answers, divergence_ratio = _solver_divergence(summary)
    if (
        status is TaskQualityGateStatus.ACCEPT
        and divergence_ratio > config.calibration.max_divergence_ratio
    ):
        status = TaskQualityGateStatus.REJECT_TOO_HARD

    return TaskQualityGateSummary(
        status=status,
        pass_rate=summary.pass_rate,
        matched_solver_runs=summary.matched_solver_runs,
        total_solver_runs=summary.total_solver_runs,
        ci_lower=interval.lower,
        ci_upper=interval.upper,
        band_lower=band.lower,
        band_upper=band.upper,
        unique_answers=unique_answers,
        divergence_ratio=divergence_ratio,
    )
