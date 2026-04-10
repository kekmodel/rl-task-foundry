"""Rolling orchestration for task-spec driven runs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rl_task_foundry.calibration.banding import PassRateBand
from rl_task_foundry.calibration.runner import calibration_decision, compute_pass_rate
from rl_task_foundry.config.models import AppConfig, ProviderConfig, SolverModelConfig
from rl_task_foundry.infra.budget import BudgetLedger
from rl_task_foundry.infra.checkpoint import ensure_checkpoint
from rl_task_foundry.infra.db import DatabasePools
from rl_task_foundry.infra.storage import (
    append_budget_ledger_entry,
    bootstrap_run_db,
    clear_budget_reservation,
    connect_run_db,
    record_budget_reservation,
    record_accepted_example,
    record_event,
    record_run,
    record_task,
    record_verification_result,
)
from rl_task_foundry.pipeline.manifest import config_hash
from rl_task_foundry.pipeline.provider_resilience import ProviderCircuitBreaker
from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.path_catalog import PathCatalog, PathSpec, build_path_catalog
from rl_task_foundry.solver.backend_openai_agents import OpenAIAgentsSolverBackend
from rl_task_foundry.solver.runtime import AgentRuntime
from rl_task_foundry.tasks.composer import ComposeRequest, TaskComposer
from rl_task_foundry.tasks.factory import TierATaskFactory
from rl_task_foundry.tasks.models import (
    AcceptedExample,
    PresentedToolBundle,
    SolverResult,
    TaskPackage,
    TaskSpec,
    VerifyResult,
)
from rl_task_foundry.tools.compiler import compile_canonical_tool_bundle, compile_path_tools
from rl_task_foundry.tools.models import ToolBundle, ToolSpec
from rl_task_foundry.tools.openai_agents_adapter import ToolExecutor
from rl_task_foundry.tools.text_utils import (
    count_phrase_reference,
    count_unit_hint_for_identifier,
    humanize_identifier,
    singularize_token,
)
from rl_task_foundry.truth.generator import TierAGroundTruthGenerator, _prepare_asyncpg_query
from rl_task_foundry.truth.schemas import GroundTruth
from rl_task_foundry.verification.policies import VerificationPolicy
from rl_task_foundry.verification.scorer import VerificationEngine

RuntimeFactory = Callable[
    [SolverModelConfig, ProviderConfig, list[ToolSpec], dict[str, ToolExecutor]],
    AgentRuntime,
]
SolverAssignment = tuple[SolverModelConfig, ProviderConfig, int]


@dataclass(slots=True)
class TaskExecutionOutcome:
    task: TaskSpec
    package: TaskPackage
    ground_truth: GroundTruth
    solver_results: list[SolverResult]
    verification_results: list[VerifyResult]
    pass_rate: float
    accepted: bool
    calibration_band: tuple[float, float]
    calibration_decision: str
    executed_solver_replicas: int
    planned_solver_replicas: int
    calibration_attempts: int = 1
    difficulty_history: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class RunSummary:
    run_id: str
    total_tasks: int
    accepted_tasks: int
    rejected_tasks: int
    skipped_tasks: int
    verification_results: int
    accepted_jsonl_path: Path
    rejected_jsonl_path: Path


@dataclass(slots=True)
class ReviewArtifact:
    task: TaskSpec
    path: PathSpec
    package: TaskPackage
    canonical_bundle: ToolBundle
    ground_truth: GroundTruth
    question_context: dict[str, object]


@dataclass(slots=True)
class _InFlightTask:
    task: TaskSpec
    reservation_id: str
    future: asyncio.Task[TaskExecutionOutcome]


@dataclass(slots=True)
class Orchestrator:
    """Top-level pipeline orchestrator for task-spec driven runs."""

    config: AppConfig
    runtime_factory: RuntimeFactory | None = None
    _graph: SchemaGraph | None = field(default=None, init=False, repr=False)
    _catalog: PathCatalog | None = field(default=None, init=False, repr=False)
    _pools: DatabasePools | None = field(default=None, init=False, repr=False)
    _provider_semaphores: dict[str, asyncio.Semaphore] | None = field(
        default=None, init=False, repr=False
    )
    _provider_circuits: dict[str, ProviderCircuitBreaker] | None = field(
        default=None, init=False, repr=False
    )
    _task_factory_impl: TierATaskFactory | None = field(default=None, init=False, repr=False)

    async def run_tasks(self, tasks: Sequence[TaskSpec]) -> RunSummary:
        run_id = f"run_{uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()
        config_fingerprint = config_hash(self.config)
        accepted_examples: list[AcceptedExample] = []
        rejected_payloads: list[dict[str, object]] = []
        processed_task_count = 0
        accepted_task_count = 0
        rejected_task_count = 0
        skipped_task_count = 0
        checkpoint = ensure_checkpoint(self.config.output.run_db_path)
        checkpoint_namespace = f"tasks:{config_fingerprint}"
        budget = BudgetLedger(
            max_run_usd=self.config.budget.max_run_usd,
            max_gpu_hours=self.config.budget.max_gpu_hours,
        )

        bootstrap_run_db(self.config.output.run_db_path)
        with connect_run_db(self.config.output.run_db_path) as conn:
            record_run(
                conn,
                run_id=run_id,
                config_hash=config_fingerprint,
                created_at=created_at,
            )
            record_event(
                conn,
                run_id=run_id,
                event_type="run_started",
                payload={"task_count": len(tasks)},
            )
            conn.commit()

            task_iter = iter(tasks)
            in_flight: dict[asyncio.Task[TaskExecutionOutcome], _InFlightTask] = {}
            budget_blocked = False
            task_window = self._task_concurrency_limit(len(tasks))

            while True:
                while not budget_blocked and len(in_flight) < task_window:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        break
                    if checkpoint.is_processed(task.task_id, namespace=checkpoint_namespace):
                        skipped_task_count += 1
                        record_task(
                            conn,
                            run_id=run_id,
                            task_id=task.task_id,
                            status="skipped_checkpoint",
                            payload=task.model_dump(mode="json"),
                        )
                        record_event(
                            conn,
                            run_id=run_id,
                            event_type="task_skipped_checkpoint",
                            payload={"task_id": task.task_id},
                        )
                        continue

                    try:
                        reservation_id = budget.reserve(
                            compose_api_usd=self._per_task_compose_budget(len(tasks)),
                            solve_api_usd=self._per_task_solve_budget(len(tasks)),
                            metadata={"task_id": task.task_id, "run_id": run_id},
                        )
                    except ValueError as exc:
                        skipped_task_count += 1
                        budget_blocked = True
                        record_task(
                            conn,
                            run_id=run_id,
                            task_id=task.task_id,
                            status="skipped_budget",
                            payload=task.model_dump(mode="json"),
                        )
                        record_event(
                            conn,
                            run_id=run_id,
                            event_type="run_budget_blocked",
                            payload={"task_id": task.task_id, "reason": str(exc)},
                        )
                        conn.commit()
                        break

                    record_budget_reservation(
                        conn,
                        reservation_id=reservation_id,
                        payload={
                            "run_id": run_id,
                            "task_id": task.task_id,
                            "compose_api_usd": self._per_task_compose_budget(len(tasks)),
                            "solve_api_usd": self._per_task_solve_budget(len(tasks)),
                        },
                    )
                    future = asyncio.create_task(self._execute_task(task))
                    in_flight[future] = _InFlightTask(
                        task=task,
                        reservation_id=reservation_id,
                        future=future,
                    )

                if not in_flight:
                    break

                done, _ = await asyncio.wait(
                    in_flight.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for future in done:
                    in_flight_task = in_flight.pop(future)
                    task = in_flight_task.task
                    reservation_id = in_flight_task.reservation_id
                    try:
                        outcome = future.result()
                    except Exception:
                        budget.release(reservation_id)
                        clear_budget_reservation(conn, reservation_id=reservation_id)
                        record_event(
                            conn,
                            run_id=run_id,
                            event_type="task_failed",
                            payload={"task_id": task.task_id},
                        )
                        await self._cancel_inflight(
                            in_flight,
                            budget=budget,
                            conn=conn,
                            run_id=run_id,
                        )
                        conn.commit()
                        raise

                    if outcome.calibration_decision.startswith("skipped_"):
                        skipped_task_count += 1
                        record_task(
                            conn,
                            run_id=run_id,
                            task_id=outcome.task.task_id,
                            status=outcome.calibration_decision,
                            payload=outcome.task.model_dump(mode="json"),
                        )
                        record_event(
                            conn,
                            run_id=run_id,
                            event_type=outcome.calibration_decision,
                            payload={
                                "task_id": outcome.task.task_id,
                                "question_source": outcome.task.question_source,
                            },
                        )
                        settled = budget.settle(
                            reservation_id,
                            compose_api_usd=self._per_task_compose_budget(len(tasks)),
                            solve_api_usd=0.0,
                        )
                        if settled.compose_api_usd:
                            append_budget_ledger_entry(
                                conn,
                                phase="compose",
                                amount_usd=settled.compose_api_usd,
                                kind="settled",
                            )
                        clear_budget_reservation(conn, reservation_id=reservation_id)
                        checkpoint.mark_processed(
                            task.task_id,
                            namespace=checkpoint_namespace,
                            payload={
                                "run_id": run_id,
                                "accepted": False,
                                "skipped": True,
                                "reason": outcome.calibration_decision,
                            },
                        )
                        checkpoint.flush_to_connection(conn)
                        continue

                    processed_task_count += 1
                    task_payload = outcome.task.model_dump(mode="json")
                    task_payload["presented_tool_bundle_id"] = outcome.package.presented_tool_bundle.bundle_id
                    record_task(
                        conn,
                        run_id=run_id,
                        task_id=outcome.task.task_id,
                        status="accepted" if outcome.accepted else "rejected",
                        payload=task_payload,
                    )
                    for verification in outcome.verification_results:
                        record_verification_result(
                            conn,
                            run_id=run_id,
                            task_id=outcome.task.task_id,
                            solver_id=verification.solver_id,
                            payload=verification.model_dump(mode="json"),
                        )
                    if outcome.accepted:
                        accepted_task_count += 1
                        accepted_example = self._build_accepted_example(outcome)
                        accepted_examples.append(accepted_example)
                        record_accepted_example(
                            conn,
                            run_id=run_id,
                            task_id=outcome.task.task_id,
                            payload=accepted_example.model_dump(mode="json"),
                        )
                    else:
                        rejected_task_count += 1
                        rejected_payloads.append(self._rejected_payload(outcome))
                    record_event(
                        conn,
                        run_id=run_id,
                        event_type="task_completed",
                        payload={
                            "task_id": outcome.task.task_id,
                            "accepted": outcome.accepted,
                            "pass_rate": outcome.pass_rate,
                            "calibration_decision": outcome.calibration_decision,
                            "calibration_attempts": outcome.calibration_attempts,
                            "executed_solver_replicas": outcome.executed_solver_replicas,
                            "planned_solver_replicas": outcome.planned_solver_replicas,
                        },
                    )
                    settled = budget.settle(
                        reservation_id,
                        compose_api_usd=self._per_task_compose_budget(len(tasks)),
                        solve_api_usd=self._actual_task_solve_budget(
                            task_count=len(tasks),
                            executed_solver_replicas=outcome.executed_solver_replicas,
                            planned_solver_replicas=outcome.planned_solver_replicas,
                        ),
                    )
                    if settled.compose_api_usd:
                        append_budget_ledger_entry(
                            conn,
                            phase="compose",
                            amount_usd=settled.compose_api_usd,
                            kind="settled",
                        )
                    if settled.solve_api_usd:
                        append_budget_ledger_entry(
                            conn,
                            phase="solve",
                            amount_usd=settled.solve_api_usd,
                            kind="settled",
                        )
                    clear_budget_reservation(conn, reservation_id=reservation_id)
                    checkpoint.mark_processed(
                        task.task_id,
                        namespace=checkpoint_namespace,
                        payload={
                            "run_id": run_id,
                            "accepted": outcome.accepted,
                        },
                    )
                    checkpoint.flush_to_connection(conn)
            record_event(
                conn,
                run_id=run_id,
                event_type="run_completed",
                payload={
                    "processed_tasks": processed_task_count,
                    "accepted_tasks": accepted_task_count,
                    "rejected_tasks": rejected_task_count,
                    "skipped_tasks": skipped_task_count,
                },
            )
            conn.commit()

        self._write_jsonl(
            self.config.output.accepted_jsonl_path,
            [example.model_dump(mode="json") for example in accepted_examples],
        )
        self._write_jsonl(self.config.output.rejected_jsonl_path, rejected_payloads)
        await self._close_resources()
        return RunSummary(
            run_id=run_id,
            total_tasks=processed_task_count + skipped_task_count,
            accepted_tasks=accepted_task_count,
            rejected_tasks=rejected_task_count,
            skipped_tasks=skipped_task_count,
            verification_results=sum(len(example.verification_results) for example in accepted_examples)
            + sum(len(payload["verification_results"]) for payload in rejected_payloads),
            accepted_jsonl_path=self.config.output.accepted_jsonl_path,
            rejected_jsonl_path=self.config.output.rejected_jsonl_path,
        )

    async def _execute_task(self, task: TaskSpec) -> TaskExecutionOutcome:
        graph, catalog = await self._graph_and_catalog()
        band = PassRateBand(
            lower=self.config.calibration.lower_pass_rate,
            upper=self.config.calibration.upper_pass_rate,
        )
        attempted_task_ids: set[str] = set()
        attempted_path_ids: set[str] = set()
        difficulty_history: list[dict[str, object]] = []
        current_task = self._normalize_task_difficulty(task, catalog)
        best_outcome: TaskExecutionOutcome | None = None

        for attempt_index in range(1, self.config.task_composer.max_attempts_per_anchor + 1):
            attempted_task_ids.add(current_task.task_id)
            attempted_path_ids.add(current_task.selected_path_id)
            outcome = await self._execute_task_attempt(current_task, graph, catalog)
            history_entry = {
                "attempt": attempt_index,
                "path_id": outcome.task.selected_path_id,
                "question_family": outcome.task.question_family,
                "outcome_type": outcome.task.outcome_type,
                "answer_fields": [field.name for field in outcome.task.answer_schema.fields],
                "required_hops": outcome.task.required_hops,
                "pass_rate": outcome.pass_rate,
                "decision": outcome.calibration_decision,
            }
            difficulty_history.append(history_entry)
            if outcome.calibration_decision.startswith("skipped_"):
                return replace(
                    outcome,
                    calibration_attempts=attempt_index,
                    difficulty_history=list(difficulty_history),
                )
            best_outcome = self._select_best_outcome(best_outcome, outcome, band)
            if outcome.accepted:
                return replace(
                    outcome,
                    calibration_attempts=attempt_index,
                    difficulty_history=list(difficulty_history),
                )

            next_task = await self._next_difficulty_task(
                current_task,
                decision=outcome.calibration_decision,
                graph=graph,
                catalog=catalog,
                attempted_path_ids=attempted_path_ids,
                attempted_task_ids=attempted_task_ids,
            )
            if next_task is None:
                break
            current_task = next_task

        if best_outcome is None:
            raise RuntimeError("adaptive execution did not produce any task outcome")
        final_decision = best_outcome.calibration_decision
        if not best_outcome.accepted:
            final_decision = f"best_so_far::{final_decision}"
        return replace(
            best_outcome,
            calibration_decision=final_decision,
            calibration_attempts=len(difficulty_history),
            difficulty_history=list(difficulty_history),
        )

    async def _execute_task_attempt(
        self,
        task: TaskSpec,
        graph: SchemaGraph,
        catalog: PathCatalog,
    ) -> TaskExecutionOutcome:
        artifact = await self.build_review_artifact(task, graph=graph, catalog=catalog)
        path = artifact.path
        ground_truth = artifact.ground_truth
        package = artifact.package
        canonical_bundle = artifact.canonical_bundle
        task = package.task
        if task.question_source == "seed_fallback":
            return TaskExecutionOutcome(
                task=task,
                package=package,
                ground_truth=ground_truth,
                solver_results=[],
                verification_results=[],
                pass_rate=0.0,
                accepted=False,
                calibration_band=(
                    self.config.calibration.lower_pass_rate,
                    self.config.calibration.upper_pass_rate,
                ),
                calibration_decision="skipped_seed_fallback",
                executed_solver_replicas=0,
                planned_solver_replicas=0,
            )
        tool_specs = self._materialize_presented_tool_specs(
            package.presented_tool_bundle,
            canonical_bundle,
            graph=graph,
            catalog=catalog,
            label_tier=task.label_tier,
        )
        tool_executors = await self._tool_executors(tool_specs)
        solver_results, verification_results, decision, planned_replicas = await self._run_calibrated_solvers(
            task,
            ground_truth,
            tool_specs,
            tool_executors,
        )
        pass_rate = compute_pass_rate(verification_results)
        band = PassRateBand(
            lower=self.config.calibration.lower_pass_rate,
            upper=self.config.calibration.upper_pass_rate,
        )
        accepted = decision == "accept"
        if decision == "continue":
            accepted = band.contains(pass_rate)
            if accepted:
                decision = "accept"
            elif pass_rate < band.lower:
                decision = "reject_too_hard"
            else:
                decision = "reject_too_easy"
        return TaskExecutionOutcome(
            task=task,
            package=package,
            ground_truth=ground_truth,
            solver_results=solver_results,
            verification_results=verification_results,
            pass_rate=pass_rate,
            accepted=accepted,
            calibration_band=(band.lower, band.upper),
            calibration_decision=decision,
            executed_solver_replicas=len(solver_results),
            planned_solver_replicas=planned_replicas,
        )

    async def build_review_artifact(
        self,
        task: TaskSpec,
        *,
        graph: SchemaGraph | None = None,
        catalog: PathCatalog | None = None,
    ) -> ReviewArtifact:
        if graph is None or catalog is None:
            graph, catalog = await self._graph_and_catalog()
        normalized_task = self._normalize_task_difficulty(task, catalog)
        path = catalog.get(normalized_task.selected_path_id)
        ground_truth = await self._ground_truth_generator(graph, catalog).generate(normalized_task)
        question_context = self._question_context(normalized_task, path, ground_truth)
        package, canonical_bundle = await self._compose_task_package(
            normalized_task,
            graph,
            path,
            question_context=question_context,
        )
        return ReviewArtifact(
            task=normalized_task,
            path=path,
            package=package,
            canonical_bundle=canonical_bundle,
            ground_truth=ground_truth,
            question_context=question_context,
        )

    async def load_graph_and_catalog(self) -> tuple[SchemaGraph, PathCatalog]:
        return await self._graph_and_catalog()

    async def aclose(self) -> None:
        await self._close_resources()

    async def _graph_and_catalog(self) -> tuple[SchemaGraph, PathCatalog]:
        if self._graph is not None and self._catalog is not None:
            return self._graph, self._catalog
        introspector = PostgresSchemaIntrospector(
            database=self.config.database,
            default_visibility=self.config.privacy.default_visibility,
            visibility_overrides=self.config.privacy.visibility_overrides,
        )
        self._graph = await introspector.introspect()
        self._catalog = build_path_catalog(self._graph, max_hops=self.config.tool_compiler.max_hops)
        return self._graph, self._catalog

    async def _pools_for_tools(self) -> DatabasePools:
        if self._pools is None:
            self._pools = await DatabasePools.create(self.config.database)
        return self._pools

    async def _provider_limits(self) -> dict[str, asyncio.Semaphore]:
        if self._provider_semaphores is None:
            self._provider_semaphores = {
                provider_name: asyncio.Semaphore(provider.max_concurrency)
                for provider_name, provider in self.config.providers.items()
            }
        return self._provider_semaphores

    def _provider_breakers(self) -> dict[str, ProviderCircuitBreaker]:
        if self._provider_circuits is None:
            self._provider_circuits = {
                provider_name: ProviderCircuitBreaker(
                    provider_name=provider_name,
                    window_s=self.config.provider_resilience.circuit_breaker_window_s,
                    threshold=self.config.provider_resilience.circuit_breaker_threshold,
                    probe_interval_s=self.config.provider_resilience.probe_interval_s,
                )
                for provider_name in self.config.providers
            }
        return self._provider_circuits

    def _provider_available(self, provider_name: str) -> bool:
        return self._provider_breakers()[provider_name].is_available()

    async def _compose_task_package(
        self,
        task: TaskSpec,
        graph: SchemaGraph,
        path: PathSpec,
        *,
        question_context: dict[str, object] | None = None,
    ) -> tuple[TaskPackage, ToolBundle]:
        canonical_bundle = compile_canonical_tool_bundle(
            graph,
            path,
            label_tier=task.label_tier,
            max_list_cardinality=self.config.tool_compiler.max_list_cardinality,
            allow_aggregates=self.config.tool_compiler.allow_aggregates,
            allow_timelines=self.config.tool_compiler.allow_timelines,
            float_precision=self.config.verification.float_precision,
            business_alias_overrides=self.config.tool_compiler.business_alias_overrides,
        )
        normalized_task = task.model_copy(
            update={
                "tool_bundle_id": canonical_bundle.bundle_id,
                "required_hops": path.hop_count,
            }
        )
        fallback_presented_bundle = (
            compile_path_tools(
                graph,
                path,
                tool_level=2,
                label_tier=task.label_tier,
                max_list_cardinality=self.config.tool_compiler.max_list_cardinality,
                allow_aggregates=self.config.tool_compiler.allow_aggregates,
                allow_timelines=self.config.tool_compiler.allow_timelines,
                float_precision=self.config.verification.float_precision,
                business_alias_overrides=self.config.tool_compiler.business_alias_overrides,
            )
            if task.tool_level == 2
            else None
        )
        composer_provider = self.config.providers[self.config.models.composer.provider]
        composer = TaskComposer(
            domain=self.config.domain,
            provider=composer_provider,
            model_ref=self.config.models.composer,
            question_temperature=self.config.task_composer.question_temperature,
            question_validation_temperature=self.config.task_composer.question_validation_temperature,
            naming_temperature_l2=self.config.tool_compiler.naming_temperature_l2,
        )
        package = await composer.compose(
            ComposeRequest(
                graph=graph,
                task=normalized_task,
                path=path,
                canonical_bundle=canonical_bundle,
                question_context=question_context or {},
                fallback_presented_bundle=fallback_presented_bundle,
            )
        )
        return package, canonical_bundle

    def _materialize_presented_tool_specs(
        self,
        presented_bundle: PresentedToolBundle,
        canonical_bundle: ToolBundle,
        *,
        graph: SchemaGraph | None = None,
        catalog: PathCatalog | None = None,
        label_tier: str | None = None,
    ) -> list[ToolSpec]:
        canonical_by_semantic_key = {tool.semantic_key: tool for tool in canonical_bundle.tools}
        materialized: list[ToolSpec] = []
        for presented_tool in presented_bundle.tools:
            canonical_tool = canonical_by_semantic_key.get(presented_tool.semantic_key)
            if canonical_tool is None:
                if graph is None or catalog is None or label_tier is None:
                    raise KeyError(
                        f"presented tool semantic_key not found in canonical bundle: {presented_tool.semantic_key}"
                    )
                path_id = presented_tool.semantic_key.split(":", 1)[0]
                distractor_path = catalog.get(path_id)
                distractor_bundle = compile_canonical_tool_bundle(
                    graph,
                    distractor_path,
                    label_tier=label_tier,
                    max_list_cardinality=self.config.tool_compiler.max_list_cardinality,
                    allow_aggregates=self.config.tool_compiler.allow_aggregates,
                    allow_timelines=self.config.tool_compiler.allow_timelines,
                    float_precision=self.config.verification.float_precision,
                    business_alias_overrides=self.config.tool_compiler.business_alias_overrides,
                )
                canonical_by_semantic_key.update(
                    {tool.semantic_key: tool for tool in distractor_bundle.tools}
                )
                canonical_tool = canonical_by_semantic_key.get(presented_tool.semantic_key)
                if canonical_tool is None:
                    raise KeyError(
                        f"presented tool semantic_key not found in canonical or distractor bundles: "
                        f"{presented_tool.semantic_key}"
                    )
            materialized.append(
                ToolSpec(
                    name=presented_tool.name,
                    description=presented_tool.description,
                    sql_template=canonical_tool.sql_template,
                    parameters=canonical_tool.parameters,
                    output_fields=canonical_tool.output_fields,
                    path_id=canonical_tool.path_id,
                    kind=canonical_tool.kind,
                    tool_level=presented_bundle.tool_level,
                    semantic_key=canonical_tool.semantic_key,
                    name_source=presented_tool.name_source,
                )
            )
        return materialized

    async def _tool_executors(
        self,
        tool_specs: Sequence[ToolSpec],
    ) -> dict[str, ToolExecutor]:
        pools = await self._pools_for_tools()
        return {
            spec.name: self._sql_tool_executor(spec, pools)
            for spec in tool_specs
        }

    def _sql_tool_executor(self, spec: ToolSpec, pools: DatabasePools) -> ToolExecutor:
        async def _execute(kwargs: dict[str, object]) -> object:
            params = {
                parameter.name: _coerce_tool_parameter(kwargs.get(parameter.name), parameter.json_type)
                for parameter in spec.parameters
            }
            sql, args = _prepare_asyncpg_query(spec.sql_template, params)
            async with pools.solver_connection() as conn:
                rows = await conn.fetch(sql, *args)
            payload_rows = [dict(row) for row in rows]
            if spec.kind in {"count", "exists", "aggregate"}:
                if not payload_rows:
                    return {field: None for field in spec.output_fields}
                return payload_rows[0]
            return payload_rows

        return _execute

    async def _run_solvers(
        self,
        task: TaskSpec,
        tool_specs: list[ToolSpec],
        tool_executors: dict[str, ToolExecutor],
    ) -> list[SolverResult]:
        provider_semaphores = await self._provider_limits()
        solver_calls = []
        for solver_config in self.config.models.solvers:
            provider_config = self.config.providers[solver_config.provider]
            for replica_index in range(solver_config.replicas):
                runtime = self._runtime_for_solver(
                    solver_config,
                    provider_config,
                    tool_specs,
                    tool_executors,
                )
                solver_calls.append(
                    self._run_solver_replica(
                        runtime=runtime,
                        provider_semaphore=provider_semaphores[solver_config.provider],
                        task=task,
                        replica_index=replica_index,
                    )
                )
        return list(await asyncio.gather(*solver_calls))

    async def _run_calibrated_solvers(
        self,
        task: TaskSpec,
        ground_truth: GroundTruth,
        tool_specs: list[ToolSpec],
        tool_executors: dict[str, ToolExecutor],
    ) -> tuple[list[SolverResult], list[VerifyResult], str, int]:
        assignments = self._replica_assignments()
        planned_replicas = len(assignments)
        if not assignments:
            return [], [], "reject_too_hard", 0
        next_replica_indices = self._next_replica_indices()

        band = PassRateBand(
            lower=self.config.calibration.lower_pass_rate,
            upper=self.config.calibration.upper_pass_rate,
        )
        canary_count = min(self.config.calibration.canary_replica_count, planned_replicas)
        solver_results: list[SolverResult] = []
        verification_results: list[VerifyResult] = []

        canary_assignments = self._rebalance_assignments(
            assignments[:canary_count],
            next_replica_indices=next_replica_indices,
        )
        if not canary_assignments:
            return [], [], "reject_provider_unavailable", planned_replicas
        canary_results = await self._run_solver_assignment_batch(
            task,
            canary_assignments,
            tool_specs,
            tool_executors,
        )
        solver_results.extend(canary_results)
        verification_results.extend(self._verify(task, ground_truth, canary_results))
        decision = calibration_decision(
            total_replicas=planned_replicas,
            results=verification_results,
            band=band,
            ci_alpha=self.config.calibration.ci_alpha,
        )
        if decision != "continue":
            return solver_results, verification_results, decision, planned_replicas

        post_canary_batch_size = max(1, self.config.calibration.post_canary_batch_size)
        remaining_assignments = assignments[canary_count:]
        for batch_start in range(0, len(remaining_assignments), post_canary_batch_size):
            assignment_batch = remaining_assignments[batch_start : batch_start + post_canary_batch_size]
            next_assignments = self._rebalance_assignments(
                assignment_batch,
                next_replica_indices=next_replica_indices,
            )
            if not next_assignments:
                return solver_results, verification_results, "reject_provider_unavailable", planned_replicas
            batch_results = await self._run_solver_assignment_batch(
                task,
                next_assignments,
                tool_specs,
                tool_executors,
            )
            solver_results.extend(batch_results)
            verification_results.extend(self._verify(task, ground_truth, batch_results))
            decision = calibration_decision(
                total_replicas=planned_replicas,
                results=verification_results,
                band=band,
                ci_alpha=self.config.calibration.ci_alpha,
            )
            if decision != "continue":
                break

        return solver_results, verification_results, decision, planned_replicas

    def _runtime_for_solver(
        self,
        solver_config: SolverModelConfig,
        provider_config: ProviderConfig,
        tool_specs: list[ToolSpec],
        tool_executors: dict[str, ToolExecutor],
    ) -> AgentRuntime:
        if self.runtime_factory is not None:
            return self.runtime_factory(
                solver_config,
                provider_config,
                tool_specs,
                tool_executors,
            )
        if solver_config.backend != "openai_agents":
            raise NotImplementedError(f"Unsupported solver backend: {solver_config.backend}")
        return OpenAIAgentsSolverBackend(
            solver_config=solver_config,
            provider_config=provider_config,
            runtime_config=self.config.solver_runtime,
            tool_specs=tool_specs,
            tool_executors=tool_executors,
            session_db_path=self.config.output.traces_dir / "sessions.sqlite",
            traces_dir=self.config.output.traces_dir,
        )

    async def _run_solver_replica(
        self,
        *,
        runtime: AgentRuntime,
        solver_config: SolverModelConfig,
        provider_config: ProviderConfig,
        provider_semaphore: asyncio.Semaphore,
        task: TaskSpec,
        replica_index: int,
    ) -> SolverResult:
        provider_breaker = self._provider_breakers()[solver_config.provider]
        if not provider_breaker.is_available():
            return self._provider_failure_result(
                task=task,
                solver_config=solver_config,
                replica_index=replica_index,
                status="provider_unavailable",
                termination_reason="provider_circuit_open",
                message="provider circuit breaker is open",
                metadata={
                    "provider": solver_config.provider,
                    "cooldown_remaining_s": provider_breaker.snapshot().cooldown_remaining_s,
                },
            )
        async with provider_semaphore:
            try:
                result = await runtime.run(task, replica_index=replica_index)
            except Exception as exc:
                provider_breaker.record_failure()
                status = "provider_timeout" if isinstance(exc, TimeoutError) else "provider_error"
                termination_reason = (
                    "provider_timeout" if isinstance(exc, TimeoutError) else "provider_runtime_error"
                )
                return self._provider_failure_result(
                    task=task,
                    solver_config=solver_config,
                    replica_index=replica_index,
                    status=status,
                    termination_reason=termination_reason,
                    message=f"{type(exc).__name__}: {exc}",
                    metadata={
                        "provider": solver_config.provider,
                        "provider_type": provider_config.type,
                        "error_type": type(exc).__name__,
                    },
                )
            provider_breaker.record_success()
            return result

    async def _run_solver_assignment_batch(
        self,
        task: TaskSpec,
        assignments: Sequence[SolverAssignment],
        tool_specs: list[ToolSpec],
        tool_executors: dict[str, ToolExecutor],
    ) -> list[SolverResult]:
        provider_semaphores = await self._provider_limits()
        calls = []
        for solver_config, provider_config, replica_index in assignments:
            runtime = self._runtime_for_solver(
                solver_config,
                provider_config,
                tool_specs,
                tool_executors,
            )
            calls.append(
                self._run_solver_replica(
                    runtime=runtime,
                    solver_config=solver_config,
                    provider_config=provider_config,
                    provider_semaphore=provider_semaphores[solver_config.provider],
                    task=task,
                    replica_index=replica_index,
                )
            )
        return list(await asyncio.gather(*calls))

    def _ground_truth_generator(
        self,
        graph: SchemaGraph,
        catalog: PathCatalog,
    ) -> TierAGroundTruthGenerator:
        return TierAGroundTruthGenerator(
            database=self.config.database,
            graph=graph,
            catalog=catalog,
            float_precision=self.config.verification.float_precision,
        )

    def _question_context(
        self,
        task: TaskSpec,
        path: PathSpec,
        ground_truth: GroundTruth,
    ) -> dict[str, object]:
        anchor_entity_label = humanize_identifier(singularize_token(task.anchor_table))
        target_entity_label = humanize_identifier(singularize_token(path.tables[-1]))
        answer_shape = self._answer_shape(task)
        answer_field_names = {
            field.name.lower()
            for field in task.answer_schema.fields
        }
        answer_source_columns = {
            source.split(".")[-1].lower()
            for field in task.answer_schema.fields
            for source in field.source_columns
            if source and not source.startswith("meta:")
        }
        forbidden_markers = self._forbidden_answer_markers(task, ground_truth)
        sanitized_rows: list[dict[str, object]] = []
        for row in ground_truth.row_context[:3]:
            sanitized_row: dict[str, object] = {}
            for key, value in row.items():
                normalized_key = key.lower()
                if normalized_key in answer_field_names or normalized_key in answer_source_columns:
                    continue
                if isinstance(value, (dict, list)):
                    continue
                if self._looks_like_answer_leak(value, forbidden_markers):
                    continue
                sanitized_row[key] = value
            if sanitized_row:
                sanitized_rows.append(sanitized_row)

        return {
            "language": task.language,
            "question_family": task.question_family,
            "outcome_type": task.outcome_type,
            "answer_shape": answer_shape,
            "relationship_depth": path.hop_count,
            "anchor_entity_label": anchor_entity_label,
            "target_entity_label": target_entity_label,
            "path_entity_labels": [
                humanize_identifier(singularize_token(table_name))
                for table_name in path.tables
            ],
            "family_intent": self._family_intent(task, answer_shape=answer_shape),
            "question_style_guide": self._question_style_guide(task, path),
            "answer_fields": [
                {
                    "name": field.name,
                    "type": field.type,
                    "description": field.description,
                    "label": humanize_identifier(field.name),
                }
                for field in task.answer_schema.fields
            ],
            "row_context": sanitized_rows,
            "forbidden_markers": forbidden_markers,
        }

    @staticmethod
    def _family_intent(task: TaskSpec, *, answer_shape: str) -> str:
        if task.question_family == "status_lookup":
            return "The user wants to confirm one concrete value tied to their own account, request, or situation."
        if task.question_family == "causal_chain":
            if answer_shape == "list":
                return (
                    "The user asks which downstream items, options, services, or entities apply to their own situation "
                    "through an indirect relationship."
                )
            return (
                "The user asks for the concrete downstream item, title, option, category, provider, language, or other "
                "real-world detail that applies to their own situation through an indirect relationship."
            )
        if task.question_family == "timeline_resolution":
            return "The user asks when something most recently happened or when a relevant event took place."
        if task.question_family == "aggregate_verification":
            return "The user asks for a count or total over related items, phrased like a real end-user request."
        return "The user asks for a concrete answer grounded in their own situation."

    def _question_style_guide(
        self,
        task: TaskSpec,
        path: PathSpec,
    ) -> dict[str, object]:
        anchor_label = humanize_identifier(singularize_token(task.anchor_table))
        target_label = humanize_identifier(singularize_token(path.tables[-1]))
        preferred_subject = self._preferred_subject_reference(path)
        downstream_label = self._downstream_subject_label(path)
        answer_labels = [
            humanize_identifier(field.name)
            for field in task.answer_schema.fields
        ]
        answer_concept_reference = self._answer_concept_reference(task, path)
        answer_shape = self._answer_shape(task)
        count_target_reference = self._count_target_reference(path)
        count_phrase_reference = self._count_phrase_reference(path)
        count_unit_hint = self._count_unit_hint(path)
        family_patterns = {
            "status_lookup": {
                "goal": "Ask for one concrete current or registered detail.",
                "preferred_shape": (
                    "Phrase the request directly around the answer field the user wants to know, "
                    "such as a status, city, date, or language."
                ),
                "avoid_shape": (
                    "Do not narrate the internal chain or say the value is connected, linked, or related."
                ),
            },
            "causal_chain": {
                "goal": (
                    "Ask which downstream items apply."
                    if answer_shape == "list"
                    else "Ask for the concrete downstream detail that actually applies in a practical way."
                ),
                "preferred_shape": (
                    "Ask which concrete downstream items, options, or entities are associated with the user."
                    if answer_shape == "list"
                    else (
                        "Phrase the question around the downstream thing the user actually cares about, such as an "
                        "item, title, option, language, provider, category, or destination, not around account/profile metadata "
                        "or a database traversal."
                    )
                ),
                "avoid_shape": (
                    "Do not ask for a single final value when the answer contract is a list."
                    if answer_shape == "list"
                    else (
                        "Avoid awkward compounds like 'my customer info', 'my user record', or "
                        "'the final value through a path'."
                    )
                ),
            },
            "timeline_resolution": {
                "goal": "Ask when something most recently happened or was updated.",
                "preferred_shape": (
                    "Focus on the event timing the user would naturally ask about."
                ),
                "avoid_shape": (
                    "Do not ask for an abstract timestamp field or refer to internal records."
                ),
            },
            "aggregate_verification": {
                "goal": "Ask for a count or total over concrete user-facing items.",
                "preferred_shape": (
                    "The counted items should sound like real things, people, or cases the user has, used, received, or can access. "
                    "Use a natural counting phrase for the target type instead of generic item-count wording."
                ),
                "avoid_shape": (
                    "Do not phrase the request as counting linked rows, connected fields, or schema-level objects."
                ),
            },
        }
        return {
            "preferred_subject_reference": preferred_subject,
            "anchor_reference": self._entity_reference_phrase(anchor_label),
            "downstream_reference": (
                self._entity_reference_phrase(downstream_label)
                if downstream_label is not None
                else None
            ),
            "target_reference": target_label,
            "count_target_reference": count_target_reference,
            "count_phrase_reference": count_phrase_reference,
            "count_unit_hint": count_unit_hint,
            "answer_labels": answer_labels,
            "answer_concept_reference": answer_concept_reference,
            "answer_shape": answer_shape,
            "family_patterns": family_patterns.get(task.question_family, {}),
            "banned_phrases": self._banned_question_phrases(path),
        }

    @staticmethod
    def _answer_shape(task: TaskSpec) -> str:
        if any(field.source_columns and field.source_columns[0] == "meta:count" for field in task.answer_schema.fields):
            return "count"
        if any(field.source_columns and field.source_columns[0] == "meta:exists" for field in task.answer_schema.fields):
            return "exists"
        if task.answer_schema.fields and all(field.type.startswith("list[") for field in task.answer_schema.fields):
            return "list"
        if task.question_family == "timeline_resolution":
            return "latest_scalar"
        return "scalar"

    @staticmethod
    def _preferred_subject_reference(path: PathSpec) -> str:
        self_like = {
            "customer",
            "user",
            "member",
            "account",
            "profile",
            "person",
            "patient",
            "student",
        }
        support_like = {
            "inventory",
            "mapping",
            "bridge",
            "xref",
            "link",
            "relation",
            "association",
            "detail",
            "entry",
            "item",
        }
        singular_tables = [singularize_token(table_name).lower() for table_name in path.tables]
        anchor = singular_tables[0]
        if anchor in self_like:
            for table_name in singular_tables[1:]:
                if table_name in self_like or table_name in support_like:
                    continue
                return Orchestrator._entity_reference_phrase(
                    humanize_identifier(table_name)
                )
            return "my account"
        return Orchestrator._entity_reference_phrase(
            humanize_identifier(anchor)
        )

    @staticmethod
    def _downstream_subject_label(path: PathSpec) -> str | None:
        support_like = {
            "inventory",
            "mapping",
            "bridge",
            "xref",
            "link",
            "relation",
            "association",
            "detail",
            "entry",
            "item",
        }
        singular_tables = [singularize_token(table_name).lower() for table_name in path.tables[1:]]
        for table_name in singular_tables:
            if table_name in support_like:
                continue
            return humanize_identifier(table_name)
        return None

    @staticmethod
    def _entity_reference_phrase(label: str) -> str:
        normalized = label.strip().lower()
        if normalized in {"account", "profile"}:
            return f"my {normalized}"
        if normalized.startswith("my "):
            return normalized
        return f"my {normalized}"

    @staticmethod
    def _banned_question_phrases(path: PathSpec) -> list[str]:
        banned = [
            "connected",
            "linked",
            "related value",
            "related item",
            "through the path",
            "customer info",
            "user info",
            "member info",
            "account record",
            "profile record",
            "연결된",
            "관련된",
            "경로를 따라",
            "고객 정보",
            "사용자 정보",
            "회원 정보",
        ]
        anchor_label = humanize_identifier(singularize_token(path.root_table)).lower()
        if anchor_label in {"customer", "user", "member", "account", "profile"}:
            banned.extend(
                [
                    f"my {anchor_label} info",
                    f"my {anchor_label} record",
                    f"제 {anchor_label}",
                ]
            )
        return sorted(set(banned))

    @staticmethod
    def _count_target_reference(path: PathSpec) -> str:
        return humanize_identifier(singularize_token(path.tables[-1]))

    @staticmethod
    def _count_phrase_reference(path: PathSpec) -> str:
        return count_phrase_reference(path.tables[-1])

    @staticmethod
    def _count_unit_hint(path: PathSpec) -> str:
        return count_unit_hint_for_identifier(path.tables[-1])

    @staticmethod
    def _answer_concept_reference(task: TaskSpec, path: PathSpec) -> str | None:
        if len(task.answer_schema.fields) != 1:
            return None
        field = task.answer_schema.fields[0]
        target_label = humanize_identifier(singularize_token(path.tables[-1]))
        normalized_field = field.name.strip().lower()
        normalized_target = target_label.strip().lower()
        if (
            normalized_field in {"name", "title", "label", "code"}
            or normalized_field.endswith(("_name", "_title", "_label", "_code"))
            or normalized_field in {
                f"{normalized_target}_name",
                f"{normalized_target}_title",
                f"{normalized_target}_label",
                f"{normalized_target}_code",
            }
        ):
            return target_label
        return humanize_identifier(field.name)

    @staticmethod
    def _normalize_answer_value(value: object) -> str | None:
        if value is None or isinstance(value, bool):
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        return normalized

    def _forbidden_answer_markers(
        self,
        task: TaskSpec,
        ground_truth: GroundTruth,
    ) -> list[dict[str, str]]:
        answer_fields_by_name = {field.name: field for field in task.answer_schema.fields}
        markers: list[dict[str, str]] = []
        for field_name, value in ground_truth.canonical_answer.items():
            field = answer_fields_by_name.get(field_name)
            if field is None:
                continue
            normalized = self._normalize_answer_value(value)
            if normalized is None:
                continue
            match_mode = "substring"
            if field.type in {"int", "float", "date", "datetime"} or len(normalized) < 4:
                match_mode = "token"
            markers.append(
                {
                    "normalized": normalized,
                    "match_mode": match_mode,
                    "display": str(value),
                }
            )
        return markers

    @classmethod
    def _looks_like_answer_leak(
        cls,
        value: object,
        markers: Sequence[dict[str, str]],
    ) -> bool:
        normalized = cls._normalize_answer_value(value)
        if normalized is None:
            return False
        tokenized = {
            token
            for token in normalized.replace("/", " ").replace("-", " ").replace("_", " ").split()
            if token
        }
        for marker in markers:
            marker_value = marker.get("normalized", "").strip().lower()
            if not marker_value:
                continue
            if marker.get("match_mode") == "token":
                if marker_value in tokenized:
                    return True
            elif marker_value in normalized:
                return True
        return False

    def _verify(
        self,
        task: TaskSpec,
        ground_truth: GroundTruth,
        solver_results: Sequence[SolverResult],
    ) -> list[VerifyResult]:
        verifier = VerificationEngine(
            VerificationPolicy(
                require_provenance=self.config.verification.require_provenance,
                fail_on_internal_field_leak=self.config.verification.fail_on_internal_field_leak,
                float_precision=self.config.verification.float_precision,
                shadow_sample_rate=self.config.verification.shadow_sample_rate,
            )
        )
        return [verifier.verify(task, ground_truth, result) for result in solver_results]

    def _build_accepted_example(self, outcome: TaskExecutionOutcome) -> AcceptedExample:
        export_payload = {
            "task_id": outcome.task.task_id,
            "question": outcome.task.question,
            "question_source": outcome.task.question_source,
            "question_generation_metadata": outcome.task.question_generation_metadata,
            "label_tier": outcome.task.label_tier,
            "outcome_type": outcome.task.outcome_type,
            "answer_schema": outcome.task.answer_schema.model_dump(mode="json"),
            "ground_truth": {
                "canonical_answer": outcome.ground_truth.canonical_answer,
                "verification_sql": outcome.ground_truth.verification_sql,
            },
            "calibration": {
                "pass_rate": outcome.pass_rate,
                "band": list(outcome.calibration_band),
                "decision": outcome.calibration_decision,
                "attempts": outcome.calibration_attempts,
                "difficulty_history": outcome.difficulty_history,
                "executed_solver_replicas": outcome.executed_solver_replicas,
                "planned_solver_replicas": outcome.planned_solver_replicas,
            },
            "metadata": {
                "anchor_table": outcome.task.anchor_table,
                "anchor_pk_column": outcome.task.anchor_pk_column,
                "anchor_pk_value": outcome.task.anchor_pk_value,
                "tool_level": outcome.task.tool_level,
                "path_id": outcome.task.selected_path_id,
                "question_family": outcome.task.question_family,
                "language": outcome.task.language,
                "question_source": outcome.task.question_source,
            },
        }
        example = AcceptedExample(
            task=outcome.task,
            ground_truth=outcome.ground_truth,
            solver_results=outcome.solver_results,
            verification_results=outcome.verification_results,
            pass_rate=outcome.pass_rate,
            calibration_band=outcome.calibration_band,
            export_payload=export_payload,
        )
        export_payload["training_metadata"] = example.training_metadata
        if example.mean_correct_solver_turns_rounded is not None:
            export_payload["mean_correct_solver_turns_rounded"] = (
                example.mean_correct_solver_turns_rounded
            )
        return example

    def _rejected_payload(self, outcome: TaskExecutionOutcome) -> dict[str, object]:
        return {
            "task": outcome.task.model_dump(mode="json"),
            "ground_truth": outcome.ground_truth.model_dump(mode="json"),
            "solver_results": [result.model_dump(mode="json") for result in outcome.solver_results],
            "verification_results": [
                result.model_dump(mode="json") for result in outcome.verification_results
            ],
            "pass_rate": outcome.pass_rate,
            "calibration_band": list(outcome.calibration_band),
            "calibration_decision": outcome.calibration_decision,
            "calibration_attempts": outcome.calibration_attempts,
            "difficulty_history": outcome.difficulty_history,
        }

    def _write_jsonl(self, path: Path, payloads: Sequence[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str))
                handle.write("\n")

    async def _close_resources(self) -> None:
        if self._pools is not None:
            await self._pools.close()
            self._pools = None

    async def _cancel_inflight(
        self,
        in_flight: dict[asyncio.Task[TaskExecutionOutcome], _InFlightTask],
        *,
        budget: BudgetLedger,
        conn,
        run_id: str,
    ) -> None:
        if not in_flight:
            return
        pending = list(in_flight.values())
        for item in pending:
            item.future.cancel()
        await asyncio.gather(*(item.future for item in pending), return_exceptions=True)
        for item in pending:
            budget.release(item.reservation_id)
            clear_budget_reservation(conn, reservation_id=item.reservation_id)
            record_event(
                conn,
                run_id=run_id,
                event_type="task_cancelled",
                payload={"task_id": item.task.task_id},
            )
        in_flight.clear()

    def _task_concurrency_limit(self, task_count: int) -> int:
        if task_count <= 0:
            return 1
        return max(
            1,
            min(
                task_count,
                self.config.database.control_pool_size,
                self.config.models.total_solver_replicas,
            ),
        )

    def _normalize_task_difficulty(
        self,
        task: TaskSpec,
        catalog: PathCatalog,
    ) -> TaskSpec:
        path = catalog.get(task.selected_path_id)
        difficulty_features = dict(path.difficulty_features)
        difficulty_features.update(task.difficulty_features)
        difficulty_features.setdefault("condition_complexity", 1)
        difficulty_features["fanout_ambiguity"] = int(path.difficulty_features.get("shortcut_count", 0))
        return task.model_copy(
            update={
                "required_hops": path.hop_count,
                "difficulty_features": difficulty_features,
            }
        )

    def _task_factory(self) -> TierATaskFactory:
        if self._task_factory_impl is None:
            self._task_factory_impl = TierATaskFactory(
                database=self.config.database,
                domain=self.config.domain,
                task_config=self.config.task_composer,
                tool_compiler=self.config.tool_compiler,
                verification=self.config.verification,
            )
        return self._task_factory_impl

    def _select_best_outcome(
        self,
        current_best: TaskExecutionOutcome | None,
        candidate: TaskExecutionOutcome,
        band: PassRateBand,
    ) -> TaskExecutionOutcome:
        if current_best is None:
            return candidate
        current_distance = self._band_distance(current_best.pass_rate, band)
        candidate_distance = self._band_distance(candidate.pass_rate, band)
        if candidate_distance < current_distance:
            return candidate
        if candidate_distance > current_distance:
            return current_best
        if candidate.executed_solver_replicas > current_best.executed_solver_replicas:
            return candidate
        return current_best

    @staticmethod
    def _band_distance(pass_rate: float, band: PassRateBand) -> float:
        if band.contains(pass_rate):
            return 0.0
        if pass_rate < band.lower:
            return band.lower - pass_rate
        return pass_rate - band.upper

    def _adaptive_candidate_tasks(
        self,
        *,
        graph: SchemaGraph,
        catalog: PathCatalog,
        task: TaskSpec,
        current_path: PathSpec,
        compatible_paths: Sequence[PathSpec],
        attempted_task_ids: set[str],
        decision: str,
    ) -> list[TaskSpec]:
        factory = self._task_factory()
        current_path_ids = [current_path.path_id]
        compatible_path_ids = [path.path_id for path in compatible_paths]
        candidate_groups: list[list[TaskSpec]] = []

        if decision == "reject_too_easy":
            if compatible_path_ids:
                candidate_groups.append(
                    factory.candidate_tasks_for_anchor(
                        graph,
                        catalog,
                        anchor_table=task.anchor_table,
                        anchor_pk_column=task.anchor_pk_column,
                        anchor_pk_value=task.anchor_pk_value,
                        path_ids=compatible_path_ids,
                        question_families=[task.question_family],
                        outcome_types=[task.outcome_type],
                    )
                )
                candidate_groups.append(
                    factory.candidate_tasks_for_anchor(
                        graph,
                        catalog,
                        anchor_table=task.anchor_table,
                        anchor_pk_column=task.anchor_pk_column,
                        anchor_pk_value=task.anchor_pk_value,
                        path_ids=compatible_path_ids,
                        question_families=self.config.task_composer.question_families,
                        outcome_types=[task.outcome_type],
                    )
                )
            candidate_groups.append(
                factory.candidate_tasks_for_anchor(
                    graph,
                    catalog,
                    anchor_table=task.anchor_table,
                    anchor_pk_column=task.anchor_pk_column,
                    anchor_pk_value=task.anchor_pk_value,
                    path_ids=current_path_ids,
                    question_families=self.config.task_composer.question_families,
                    outcome_types=[task.outcome_type],
                )
            )
        else:
            if compatible_path_ids:
                candidate_groups.append(
                    factory.candidate_tasks_for_anchor(
                        graph,
                        catalog,
                        anchor_table=task.anchor_table,
                        anchor_pk_column=task.anchor_pk_column,
                        anchor_pk_value=task.anchor_pk_value,
                        path_ids=compatible_path_ids,
                        question_families=[task.question_family],
                        outcome_types=[task.outcome_type],
                    )
                )
                candidate_groups.append(
                    factory.candidate_tasks_for_anchor(
                        graph,
                        catalog,
                        anchor_table=task.anchor_table,
                        anchor_pk_column=task.anchor_pk_column,
                        anchor_pk_value=task.anchor_pk_value,
                        path_ids=compatible_path_ids,
                        question_families=self.config.task_composer.question_families,
                        outcome_types=[task.outcome_type],
                    )
                )
            candidate_groups.append(
                factory.candidate_tasks_for_anchor(
                    graph,
                    catalog,
                    anchor_table=task.anchor_table,
                    anchor_pk_column=task.anchor_pk_column,
                    anchor_pk_value=task.anchor_pk_value,
                    path_ids=current_path_ids,
                    question_families=self.config.task_composer.question_families,
                    outcome_types=[task.outcome_type],
                )
            )
        for candidate_group in candidate_groups:
            filtered = [
                candidate
                for candidate in candidate_group
                if candidate.task_id not in attempted_task_ids
            ]
            if filtered:
                return filtered
        return []

    def _ordered_difficulty_candidates(
        self,
        *,
        current_task: TaskSpec,
        candidate_tasks: Sequence[TaskSpec],
        decision: str,
    ) -> list[TaskSpec]:
        current_score = self._task_difficulty_score(current_task)
        if decision == "reject_too_easy":
            harder = [task for task in candidate_tasks if self._task_difficulty_score(task) > current_score]
            if not harder:
                return []
            return sorted(
                harder,
                key=lambda task: (
                    self._task_difficulty_score(task),
                    task.selected_path_id,
                    task.question_family,
                    task.task_id,
                ),
            )
        if decision == "reject_too_hard":
            easier = [task for task in candidate_tasks if self._task_difficulty_score(task) < current_score]
            if not easier:
                return []
            return sorted(
                easier,
                key=lambda task: (
                    -self._task_difficulty_score(task),
                    task.selected_path_id,
                    task.question_family,
                    task.task_id,
                ),
            )
        return []

    def _task_difficulty_score(self, task: TaskSpec) -> float:
        family_rank = self.config.calibration.difficulty_family_order.get(
            task.question_family,
            0,
        )
        outcome_rank = self.config.calibration.difficulty_outcome_order.get(
            task.outcome_type,
            0,
        )
        fanout = float(task.difficulty_features.get("fanout_product", 1.0))
        shortcuts = int(task.difficulty_features.get("shortcut_count", 0))
        answer_width = len(task.answer_schema.fields)
        return (
            float(task.required_hops) * self.config.calibration.difficulty_weight_hops
            + float(family_rank) * self.config.calibration.difficulty_weight_family
            + float(outcome_rank) * self.config.calibration.difficulty_weight_outcome
            + float(answer_width) * self.config.calibration.difficulty_weight_answer_width
            + min(fanout, self.config.calibration.difficulty_fanout_cap)
            * self.config.calibration.difficulty_weight_fanout
            + float(shortcuts) * self.config.calibration.difficulty_weight_shortcuts
        )

    async def _next_difficulty_task(
        self,
        task: TaskSpec,
        *,
        decision: str,
        graph: SchemaGraph,
        catalog: PathCatalog,
        attempted_path_ids: set[str],
        attempted_task_ids: set[str],
    ) -> TaskSpec | None:
        if task.outcome_type not in {"answer", "no_result"}:
            return None
        strategy = self._adaptive_family_strategy(task)
        if strategy != "path_compatible":
            return None

        current_path = catalog.get(task.selected_path_id)
        compatible_paths = self._compatible_adaptive_paths(
            task,
            catalog=catalog,
            current_path=current_path,
            attempted_path_ids=attempted_path_ids,
        )
        candidate_tasks = self._adaptive_candidate_tasks(
            graph=graph,
            catalog=catalog,
            task=task,
            current_path=current_path,
            compatible_paths=compatible_paths,
            attempted_task_ids=attempted_task_ids,
            decision=decision,
        )
        if not candidate_tasks:
            return None

        ordered_candidates = self._ordered_difficulty_candidates(
            current_task=task,
            candidate_tasks=candidate_tasks,
            decision=decision,
        )
        if not ordered_candidates:
            return None
        generator = self._ground_truth_generator(graph, catalog)
        next_task: TaskSpec | None = None
        for candidate in ordered_candidates:
            try:
                await generator.generate(candidate)
            except Exception:
                continue
            next_task = candidate
            break
        if next_task is None:
            return None
        axis = (
            "required_hops"
            if next_task.required_hops != task.required_hops
            else (
                "contract_rewrite"
                if next_task.question_family != task.question_family
                or next_task.outcome_type != task.outcome_type
                or [field.name for field in next_task.answer_schema.fields]
                != [field.name for field in task.answer_schema.fields]
                else "fanout_ambiguity"
            )
        )
        updated_features = dict(next_task.difficulty_features)
        updated_features["difficulty_axis"] = axis
        updated_features["fanout_ambiguity"] = int(
            updated_features.get("fanout_ambiguity", next_task.difficulty_features.get("shortcut_count", 0))
        )
        return next_task.model_copy(update={"difficulty_features": updated_features})

    @staticmethod
    def _adaptive_family_strategy(task: TaskSpec) -> str | None:
        if task.question_family in {
            "status_lookup",
            "causal_chain",
            "timeline_resolution",
            "aggregate_verification",
        }:
            return "path_compatible"
        return None

    def _compatible_adaptive_paths(
        self,
        task: TaskSpec,
        *,
        catalog: PathCatalog,
        current_path: PathSpec,
        attempted_path_ids: set[str],
    ) -> list[PathSpec]:
        source_tables = self._answer_source_tables(task, current_path)
        if source_tables is None:
            return []
        compatible: list[PathSpec] = []
        for path in catalog.for_root(current_path.root_table):
            if path.path_id == current_path.path_id or path.path_id in attempted_path_ids:
                continue
            if path.tables[-1] != current_path.tables[-1]:
                continue
            if not source_tables.issubset(set(path.tables)):
                continue
            compatible.append(path)
        return compatible

    @staticmethod
    def _answer_source_tables(task: TaskSpec, current_path: PathSpec) -> set[str] | None:
        source_tables: set[str] = set()
        for field in task.answer_schema.fields:
            if field.visibility != "user_visible":
                continue
            if field.source_columns:
                marker = field.source_columns[0].strip().lower()
                if marker in {"meta:count", "meta:exists"}:
                    return None
                parts = marker.split(".")
                if len(parts) == 2:
                    source_tables.add(parts[0])
                elif len(parts) == 3:
                    source_tables.add(parts[1])
                else:
                    return None
            else:
                source_tables.add(current_path.tables[-1])
        return source_tables

    def _replica_assignments(self) -> list[SolverAssignment]:
        assignments: list[SolverAssignment] = []
        for solver_config in self.config.models.solvers:
            provider_config = self.config.providers[solver_config.provider]
            for replica_index in range(solver_config.replicas):
                assignments.append((solver_config, provider_config, replica_index))
        full_limit = min(self.config.calibration.full_replica_limit, len(assignments))
        return assignments[:full_limit]

    def _next_replica_indices(self) -> dict[str, int]:
        return {
            solver_config.solver_id: solver_config.replicas
            for solver_config in self.config.models.solvers
        }

    def _rebalance_assignments(
        self,
        assignments: Sequence[SolverAssignment],
        *,
        next_replica_indices: dict[str, int],
    ) -> list[SolverAssignment]:
        if not assignments:
            return []

        retained: list[SolverAssignment] = []
        missing_slots = 0
        for assignment in assignments:
            solver_config, provider_config, _replica_index = assignment
            if self._provider_available(solver_config.provider):
                retained.append(assignment)
            else:
                missing_slots += 1

        if missing_slots == 0:
            return retained

        healthy_candidates = [
            solver_config
            for solver_config in self.config.models.solvers
            if self._provider_available(solver_config.provider)
        ]
        if not healthy_candidates:
            return retained

        for offset in range(missing_slots):
            solver_config = healthy_candidates[offset % len(healthy_candidates)]
            provider_config = self.config.providers[solver_config.provider]
            replica_index = next_replica_indices.get(solver_config.solver_id, solver_config.replicas)
            next_replica_indices[solver_config.solver_id] = replica_index + 1
            retained.append((solver_config, provider_config, replica_index))
        return retained

    def _provider_failure_result(
        self,
        *,
        task: TaskSpec,
        solver_config: SolverModelConfig,
        replica_index: int,
        status: str,
        termination_reason: str,
        message: str,
        metadata: dict[str, object],
    ) -> SolverResult:
        return SolverResult(
            task_id=task.task_id,
            solver_id=solver_config.solver_id,
            provider=solver_config.provider,
            model=solver_config.model,
            replica_index=replica_index,
            transcript_ref=(
                f"memory://provider_failure/{task.task_id}/{solver_config.solver_id}/replica-{replica_index}"
            ),
            tool_trace_ref=(
                f"memory://provider_failure_tools/{task.task_id}/{solver_config.solver_id}/replica-{replica_index}"
            ),
            raw_output_text=message,
            structured_output=None,
            explicit_memory_events=[],
            token_usage={},
            latency_ms=0,
            turn_count=0,
            status=status,
            termination_reason=termination_reason,
            termination_metadata=metadata,
        )

    def _per_task_compose_budget(self, task_count: int) -> float:
        if task_count <= 0:
            return 0.0
        return self.config.budget.compose_phase_usd / task_count

    def _per_task_solve_budget(self, task_count: int) -> float:
        if task_count <= 0:
            return 0.0
        return self.config.budget.solve_phase_usd / task_count

    def _actual_task_solve_budget(
        self,
        *,
        task_count: int,
        executed_solver_replicas: int,
        planned_solver_replicas: int,
    ) -> float:
        if task_count <= 0 or planned_solver_replicas <= 0:
            return 0.0
        per_task_budget = self._per_task_solve_budget(task_count)
        return per_task_budget * (executed_solver_replicas / planned_solver_replicas)


def _coerce_tool_parameter(value: object, json_type: str) -> object:
    if value is None:
        return None
    if json_type == "integer":
        return int(value)
    if json_type == "number":
        return float(value)
    if json_type == "boolean":
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"cannot coerce boolean parameter from {value!r}")
    return str(value)
