"""Task-bundle-first rollout orchestration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from pathlib import Path
from rl_task_foundry.calibration.banding import PassRateBand, clopper_pearson_interval
from rl_task_foundry.calibration.runner import calibration_decision
from rl_task_foundry.config.models import AppConfig, ProviderConfig, SolverModelConfig
from rl_task_foundry.infra.db import DatabasePools, ensure_attached_database_pools
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.solver.backend_openai_agents import OpenAIAgentsSolverBackend
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.solver.runtime import AgentRuntime, SolverEpisodeInput
from rl_task_foundry.synthesis.canonicalize import (
    RewardResult,
    RewardStatus,
    canonical_json,
    compute_reward,
)
from rl_task_foundry.synthesis.contracts import TaskBundleContract
from rl_task_foundry.synthesis.runtime import SynthesisTaskDraft
from rl_task_foundry.tooling.atomic import (
    AtomicSession,
    CursorStore,
    build_atomic_tools,
)
from rl_task_foundry.tooling.common import SchemaSnapshot, snapshot_from_graph


SdkToolsFactory = Callable[[TaskBundleContract], Awaitable[list[object]]]


TaskRuntimeFactory = Callable[
    [
        SolverModelConfig,
        ProviderConfig,
        TaskBundleContract,
        list[object],
    ],
    AgentRuntime,
]
TaskSolverRunFactory = Callable[[], Awaitable["TaskSolverRun"]]


@dataclass(frozen=True, slots=True)
class TaskRolloutBundle:
    task_bundle: TaskBundleContract
    rendered_user_prompt: str
    canonical_answer_json: str
    label_signature: str

    @classmethod
    def from_draft(cls, draft: SynthesisTaskDraft) -> "TaskRolloutBundle":
        return cls(
            task_bundle=draft.task_bundle,
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
    failed_solver_runs: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_solver_runs == 0:
            return 0.0
        return self.matched_solver_runs / self.total_solver_runs


def _evaluable_runs(runs: list["TaskSolverRun"]) -> list["TaskSolverRun"]:
    # Infrastructure failures (RateLimitError, BadRequestError, APITimeoutError,
    # etc.) surface as solver_result.status == "failed" with empty raw_output.
    # Excluding them keeps pass_rate a measure of task difficulty rather than
    # provider availability.
    return [r for r in runs if r.solver_result.status != "failed"]


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
    sdk_tools_factory: SdkToolsFactory | None = None
    database_pools: DatabasePools | None = None
    traces_dir_override: Path | None = None
    event_logger: object | None = None
    _provider_semaphores: dict[str, asyncio.Semaphore] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _schema_snapshot_cache: dict[str, SchemaSnapshot] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _snapshot_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    _database_pools: DatabasePools | None = field(default=None, init=False, repr=False)
    _owns_database_pools: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.database_pools is not None:
            self._database_pools = self.database_pools
            self._owns_database_pools = False

    async def run_draft(self, draft: SynthesisTaskDraft) -> TaskRolloutSummary:
        return await self.run_bundle(TaskRolloutBundle.from_draft(draft))

    async def run_bundle(self, bundle: TaskRolloutBundle) -> TaskRolloutSummary:
        runs: list[TaskSolverRun] = []
        calls: list[TaskSolverRunFactory] = []
        for solver_index, solver_config in enumerate(self.config.models.solvers):
            provider_config = self.config.providers[solver_config.provider]
            calls.append(
                partial(
                    self._run_solver,
                    solver_config=solver_config,
                    provider_config=provider_config,
                    solver_index=solver_index,
                    bundle=bundle,
                )
            )

        planned_solver_runs = min(len(calls), self.config.calibration.max_solver_runs)
        scheduled_calls = calls[:planned_solver_runs]
        early_stop_decision: str | None = None
        if scheduled_calls:
            runs, early_stop_decision = await self._execute_solver_batches(scheduled_calls)
        evaluable = _evaluable_runs(runs)
        matched_solver_runs = sum(1 for run in evaluable if run.reward_result.status == RewardStatus.MATCHED)
        return TaskRolloutSummary(
            task_id=bundle.task_bundle.task_id,
            db_id=bundle.task_bundle.db_id,
            planned_solver_runs=planned_solver_runs,
            total_solver_runs=len(evaluable),
            matched_solver_runs=matched_solver_runs,
            early_stop_decision=early_stop_decision,
            runs=tuple(runs),
            failed_solver_runs=len(runs) - len(evaluable),
        )

    async def close(self) -> None:
        if self._database_pools is not None:
            if self._owns_database_pools:
                await self._database_pools.close()
            self._database_pools = None
        self._schema_snapshot_cache.clear()
        OpenAIAgentsSolverBackend.clear_model_cache()

    async def _run_solver(
        self,
        *,
        solver_config: SolverModelConfig,
        provider_config: ProviderConfig,
        solver_index: int,
        bundle: TaskRolloutBundle,
    ) -> TaskSolverRun:
        episode = SolverEpisodeInput(
            task_bundle=bundle.task_bundle,
            rendered_user_prompt=bundle.rendered_user_prompt,
        )
        provider_semaphore = self._provider_semaphore(solver_config.provider)
        async with provider_semaphore:
            try:
                solver_result = await self._run_with_tools(
                    solver_config=solver_config,
                    provider_config=provider_config,
                    bundle=bundle,
                    episode=episode,
                )
            except Exception as exc:
                solver_result = SolverResult(
                    task_id=episode.task_id,
                    solver_id=solver_config.solver_id,
                    provider=solver_config.provider,
                    model=solver_config.model,
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
            canonical_answer=json.loads(bundle.canonical_answer_json),
            output_schema=bundle.task_bundle.task.output_schema,
        )
        if self.event_logger is not None:
            termination_metadata = solver_result.termination_metadata or {}
            run_items = termination_metadata.get("run_items", [])
            self.event_logger.log_sync(
                actor="solver",
                actor_id=solver_config.solver_id,
                event_type="solver_run_completed",
                payload={
                    "task_id": bundle.task_bundle.task_id,
                    "solver_index": solver_index,
                    "status": solver_result.status,
                    "termination_reason": solver_result.termination_reason,
                    "turn_count": solver_result.turn_count,
                    "latency_ms": solver_result.latency_ms,
                    "matched": reward_result.status == RewardStatus.MATCHED,
                    "raw_output_preview": solver_result.raw_output_text[:200]
                    if solver_result.raw_output_text
                    else "",
                    "run_items": run_items,
                },
            )
        return TaskSolverRun(
            task_id=bundle.task_bundle.task_id,
            solver_id=solver_config.solver_id,
            solver_index=solver_index,
            solver_result=solver_result,
            reward_result=reward_result,
        )

    async def _run_with_tools(
        self,
        *,
        solver_config: SolverModelConfig,
        provider_config: ProviderConfig,
        bundle: TaskRolloutBundle,
        episode: SolverEpisodeInput,
    ) -> SolverResult:
        if self.sdk_tools_factory is not None:
            sdk_tools = await self.sdk_tools_factory(bundle.task_bundle)
            runtime = self._runtime_for_solver(
                solver_config=solver_config,
                provider_config=provider_config,
                task_bundle=bundle.task_bundle,
                sdk_tools=sdk_tools,
            )
            return await runtime.run(episode)
        pools = await self._database_pools_for_tools()
        snapshot = await self._schema_snapshot(bundle.task_bundle.db_id)
        async with pools.solver_connection() as conn:
            session = AtomicSession(
                snapshot=snapshot, connection=conn, store=CursorStore()
            )
            sdk_tools = build_atomic_tools(session)
            runtime = self._runtime_for_solver(
                solver_config=solver_config,
                provider_config=provider_config,
                task_bundle=bundle.task_bundle,
                sdk_tools=sdk_tools,
            )
            return await runtime.run(episode)

    def _provider_semaphore(self, provider_name: str) -> asyncio.Semaphore:
        semaphore = self._provider_semaphores.get(provider_name)
        if semaphore is None:
            provider_config = self.config.providers[provider_name]
            semaphore = asyncio.Semaphore(provider_config.max_concurrency)
            self._provider_semaphores[provider_name] = semaphore
        return semaphore

    def _runtime_for_solver(
        self,
        *,
        solver_config: SolverModelConfig,
        provider_config: ProviderConfig,
        task_bundle: TaskBundleContract,
        sdk_tools: list[object],
    ) -> AgentRuntime:
        if self.runtime_factory is not None:
            return self.runtime_factory(
                solver_config,
                provider_config,
                task_bundle,
                sdk_tools,
            )
        if solver_config.backend != "openai_agents":
            raise NotImplementedError(f"Unsupported solver backend: {solver_config.backend}")
        runtime_config = self.config.solver_runtime.model_copy(
            update={"max_turns": task_bundle.rollout_constraints.max_turns}
        )
        traces_dir = self.traces_dir_override or self.config.output.traces_dir
        return OpenAIAgentsSolverBackend(
            solver_config=solver_config,
            provider_config=provider_config,
            runtime_config=runtime_config,
            sdk_tools=sdk_tools,
            session_db_path=traces_dir / "sessions.sqlite",
        )

    async def _schema_snapshot(self, db_id: str) -> SchemaSnapshot:
        cached = self._schema_snapshot_cache.get(db_id)
        if cached is not None:
            return cached
        async with self._snapshot_lock:
            cached = self._schema_snapshot_cache.get(db_id)
            if cached is not None:
                return cached
            introspector = PostgresSchemaIntrospector(
                database=self.config.database,
                default_visibility=self.config.privacy.default_visibility,
                visibility_overrides=self.config.privacy.visibility_overrides,
            )
            graph = await introspector.introspect()
            snapshot = snapshot_from_graph(graph)
            self._schema_snapshot_cache[db_id] = snapshot
            return snapshot

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
                results=[run.reward_result.status == RewardStatus.MATCHED for run in _evaluable_runs(runs)],
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


def _solver_divergence(summary: TaskRolloutSummary) -> tuple[int, float]:
    answers: set[str] = set()
    submitted = 0
    for run in summary.runs:
        text = run.solver_result.raw_output_text
        if not text:
            continue
        answers.add(_solver_divergence_key(text))
        submitted += 1
    unique = len(answers)
    ratio = unique / submitted if submitted > 0 else 0.0
    return unique, ratio


def _solver_divergence_key(text: str) -> str:
    if not text.strip().startswith(("{", "[")):
        return text
    try:
        return canonical_json(json.loads(text))
    except json.JSONDecodeError:
        return text


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
            results=[run.reward_result.status == RewardStatus.MATCHED for run in _evaluable_runs(list(summary.runs))],
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
        and config.calibration.max_divergence_ratio is not None
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
