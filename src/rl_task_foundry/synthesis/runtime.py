"""Synthesis meta-agent runtime skeleton."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Protocol

from pydantic import Field, model_validator

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.pipeline.provider_resilience import (
    ProviderCircuitBreaker,
    ProviderCircuitSnapshot,
)
from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    CrossInstanceSet,
    DifficultyAxis,
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnvironmentStatus,
    InstanceContract,
    InstanceSpaceContract,
    OutputFieldContract,
    OutputFieldType,
    RolloutConstraintsContract,
    ShadowVerifierContract,
    SolutionContract,
    StrictModel,
    TaskContract,
    VerifierContract,
)
from rl_task_foundry.synthesis.canonicalize import (
    CanonicalizationError,
    canonical_json,
    canonicalize_output,
)
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle, AtomicToolGenerator
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.registration_runner import (
    GeneratedArtifactBundle,
    RegistrationArtifactName,
    RegistrationBundleDiagnostics,
    RegistrationBundleReport,
    RegistrationBundleStatus,
    VerifierProbeSpec,
    build_registration_diagnostics,
    run_registration_bundle,
)
from rl_task_foundry.synthesis.rendered_prompt_builder import build_rendered_user_prompt
from rl_task_foundry.synthesis.subprocess_pool import RegistrationSubprocessPool
from rl_task_foundry.synthesis.subprocess_pool import RegistrationSelfConsistencyResult

CURRENT_SYNTHESIS_GENERATOR_VERSION = "milestone-3-runtime-v1"
RUNTIME_OWNED_ENVIRONMENT_FIELDS = frozenset(
    {
        "env_id",
        "db_id",
        "domain",
        "category",
        "atomic_tool_set_ref",
        "difficulty_vector",
        "created_at",
        "generator_version",
        "tool_signature",
        "task_signature",
        "verifier_signature",
        "status",
        "quality_metrics",
        "rollout_constraints",
    }
)


class SynthesisPhase(StrEnum):
    SCHEMA_EXPLORATION = "schema_exploration"
    CATEGORY_INFERENCE = "category_inference"
    ARTIFACT_GENERATION = "artifact_generation"


class SynthesisProviderStatus(StrictModel):
    observed_at: datetime
    total_requests: int
    failures: int
    error_rate: float
    available: bool
    cooldown_remaining_s: float


class SynthesisCategoryStatus(StrictModel):
    db_id: str
    category: CategoryTaxonomy
    consecutive_discards: int
    backed_off: bool
    backoff_until: datetime | None = None
    backoff_remaining_s: float = 0.0
    last_outcome: SynthesisSelfConsistencyOutcome | None = None
    last_error_codes: list[str] = Field(default_factory=list)
    last_updated_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_timezones(self) -> SynthesisCategoryStatus:
        if self.backoff_until is not None and self.backoff_until.tzinfo is None:
            raise ValueError("backoff_until must be timezone-aware")
        if self.last_updated_at is not None and self.last_updated_at.tzinfo is None:
            raise ValueError("last_updated_at must be timezone-aware")
        return self


class SynthesisMemoryEntry(StrictModel):
    phase: SynthesisPhase
    provider: str
    model: str
    summary: str
    turn_count: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    transcript_ref: str | None = None
    tool_trace_ref: str | None = None


class SynthesisToolTraceEntry(StrictModel):
    phase: SynthesisPhase
    provider: str
    model: str
    tool_name: str
    raw_repr: str = ""
    semantic_key: str | None = None


class SchemaExplorationOutput(StrictModel):
    domain_hypothesis: str
    candidate_categories: list[CategoryTaxonomy] = Field(min_length=1)
    memory_summary: str = "schema exploration completed"


class CategoryInferenceOutput(StrictModel):
    selected_category: CategoryTaxonomy
    rationale: str
    memory_summary: str = "category inference completed"


class ProposedEnvironmentDraft(StrictModel):
    task: TaskContract
    solution: SolutionContract
    verifier: VerifierContract
    shadow_verifier: ShadowVerifierContract
    instance_space: InstanceSpaceContract
    cross_instance_set: CrossInstanceSet = Field(default_factory=CrossInstanceSet)


class ArtifactGenerationOutput(StrictModel):
    proposed_environment: ProposedEnvironmentDraft
    artifacts: GeneratedArtifactBundle
    memory_summary: str = "artifact generation completed"


SynthesisPhaseOutput = (
    SchemaExplorationOutput | CategoryInferenceOutput | ArtifactGenerationOutput
)


class SynthesisStageRequest(StrictModel):
    phase: SynthesisPhase
    db_id: str
    atomic_tool_set_ref: str | None = None
    available_atomic_tools: list[dict[str, object]] = Field(default_factory=list)
    domain_name: str
    user_role: str
    agent_role: str
    scenario_description: str
    requested_category: CategoryTaxonomy | None = None
    attempt_index: int = 1
    schema_summary: dict[str, object] = Field(default_factory=dict)
    previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput] = Field(default_factory=dict)
    memory: list[SynthesisMemoryEntry] = Field(default_factory=list)
    latest_registration_diagnostics: RegistrationBundleDiagnostics | None = None
    latest_self_consistency_diagnostics: SynthesisSelfConsistencyDiagnostics | None = None
    strongest_difficulty_vector: dict[DifficultyAxis, float] = Field(default_factory=dict)
    difficulty_crank_index: int = Field(default=0, ge=0)
    difficulty_crank_history: list[DifficultyAxis] = Field(default_factory=list)


class SynthesisStageResult(StrictModel):
    phase: SynthesisPhase
    provider: str
    model: str
    payload: SynthesisPhaseOutput
    memory_entry: SynthesisMemoryEntry
    tool_traces: list[SynthesisToolTraceEntry] = Field(default_factory=list)


class MaterializedInstanceRecord(StrictModel):
    instance_id: str
    rendered_user_prompt: str
    params: dict[str, object] = Field(default_factory=dict)
    anchor_values: dict[str, object] = Field(default_factory=dict)


class MaterializedCanonicalAnswerRecord(StrictModel):
    instance_id: str
    canonical_answer: object
    canonical_answer_json: str
    solution_fingerprint: str


class SynthesisEnvironmentDraft(StrictModel):
    created_at: datetime
    db_id: str
    requested_category: CategoryTaxonomy
    schema_summary: dict[str, object] = Field(default_factory=dict)
    selected_category: CategoryTaxonomy
    environment: EnvironmentContract
    atomic_tool_bundle: AtomicToolBundle
    artifacts: GeneratedArtifactBundle
    registration_report: RegistrationBundleReport
    registration_diagnostics: RegistrationBundleDiagnostics
    self_consistency_diagnostics: SynthesisSelfConsistencyDiagnostics
    instances: list[MaterializedInstanceRecord] = Field(default_factory=list)
    canonical_answers: list[MaterializedCanonicalAnswerRecord] = Field(default_factory=list)
    self_consistency_attempts: list[SynthesisSelfConsistencyAttempt] = Field(default_factory=list)
    stage_results: list[SynthesisStageResult] = Field(default_factory=list)
    memory: list[SynthesisMemoryEntry] = Field(default_factory=list)
    tool_traces: list[SynthesisToolTraceEntry] = Field(default_factory=list)
    provider_status: dict[str, SynthesisProviderStatus] = Field(default_factory=dict)


class SynthesisStageBackend(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def run_stage(self, request: SynthesisStageRequest) -> SynthesisStageResult: ...


class SynthesisRuntimeError(RuntimeError):
    """Base runtime error for synthesis orchestration."""


@dataclass(frozen=True, slots=True)
class SynthesisBackendFailure:
    provider: str
    model: str
    error_type: str


class SynthesisPhaseExecutionError(SynthesisRuntimeError):
    """Raised when a synthesis phase fails across all candidate backends."""

    def __init__(
        self,
        message: str,
        *,
        phase: SynthesisPhase | None = None,
        backend_failures: list[SynthesisBackendFailure] | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.backend_failures = tuple(backend_failures or [])


class SynthesisProviderUnavailableError(SynthesisRuntimeError):
    """Raised when all provider candidates are currently unavailable."""

    def __init__(self, message: str, *, phase: SynthesisPhase | None = None) -> None:
        super().__init__(message)
        self.phase = phase


class SynthesisCategoryMismatchError(SynthesisRuntimeError):
    """Raised when category inference diverges from the requested category."""


class SynthesisRegistrationError(SynthesisRuntimeError):
    """Raised when generated artifacts fail registration validation."""

    def __init__(
        self,
        message: str,
        *,
        report: RegistrationBundleReport | None = None,
        diagnostics: RegistrationBundleDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.diagnostics = diagnostics


class SynthesisSelfConsistencyOutcome(StrEnum):
    PASSED = "passed"
    REGISTRATION_FAILED = "registration_failed"
    CATEGORY_MISMATCH = "category_mismatch"
    SELF_CONSISTENCY_FAILED = "self_consistency_failed"
    DIFFICULTY_WEAKENED = "difficulty_weakened"
    DIFFICULTY_CRANK_INVALID = "difficulty_crank_invalid"
    DIFFICULTY_CRANK_LIMIT_EXCEEDED = "difficulty_crank_limit_exceeded"


class SynthesisSelfConsistencyDiagnostics(StrictModel):
    passed: bool
    error_codes: list[str] = Field(default_factory=list)
    weak_signal_codes: list[str] = Field(default_factory=list)
    answer: object | None = None
    solution_tool_calls: int | None = None
    verifier_tool_calls: int | None = None
    shadow_verifier_tool_calls: int | None = None
    fetch_facts_return_keys: list[str] = Field(default_factory=list)
    expected_fact_keys: list[str] = Field(default_factory=list)
    missing_fact_keys: list[str] = Field(default_factory=list)
    extra_fact_keys: list[str] = Field(default_factory=list)
    fetch_facts_answer_reads: int | None = None
    facts_match_answer_reads: int | None = None
    facts_match_facts_reads: int | None = None
    check_constraints_facts_reads: int | None = None
    facts_match_result: bool | None = None
    check_constraints_result: bool | None = None
    verify_result: bool | None = None
    shadow_verify_result: bool | None = None


class SynthesisSelfConsistencyAttempt(StrictModel):
    attempt_index: int
    outcome: SynthesisSelfConsistencyOutcome
    provider: str
    model: str
    memory_summary: str
    error_message: str | None = None
    registration_diagnostics: RegistrationBundleDiagnostics | None = None
    self_consistency_diagnostics: SynthesisSelfConsistencyDiagnostics | None = None


class SynthesisSelfConsistencyError(SynthesisRuntimeError):
    """Raised when artifact generation exhausts its self-consistency budget."""

    def __init__(
        self,
        message: str,
        *,
        attempts: list[SynthesisSelfConsistencyAttempt],
        last_registration_diagnostics: RegistrationBundleDiagnostics | None = None,
        last_self_consistency_diagnostics: SynthesisSelfConsistencyDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_registration_diagnostics = last_registration_diagnostics
        self.last_self_consistency_diagnostics = last_self_consistency_diagnostics


class SynthesisDbBindingError(SynthesisRuntimeError):
    """Raised when a runtime instance is reused for a different logical database id."""


class SynthesisCategoryBackoffError(SynthesisRuntimeError):
    """Raised when a db/category pair is temporarily backed off after repeated discards."""

    def __init__(
        self,
        message: str,
        *,
        db_id: str,
        category: CategoryTaxonomy,
        consecutive_discards: int,
        backoff_until: datetime,
        last_outcome: SynthesisSelfConsistencyOutcome | None = None,
        last_error_codes: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.db_id = db_id
        self.category = category
        self.consecutive_discards = consecutive_discards
        self.backoff_until = backoff_until
        self.last_outcome = last_outcome
        self.last_error_codes = list(last_error_codes or [])


@dataclass(slots=True)
class _CategoryFailureState:
    consecutive_discards: int = 0
    backoff_until: datetime | None = None
    last_outcome: SynthesisSelfConsistencyOutcome | None = None
    last_error_codes: list[str] = field(default_factory=list)
    last_updated_at: datetime | None = None


SynthesisCategoryStatus.model_rebuild()
SynthesisStageRequest.model_rebuild()
SynthesisEnvironmentDraft.model_rebuild()


def summarize_schema_graph(graph: SchemaGraph, *, max_tables: int = 32) -> dict[str, object]:
    table_summaries: list[dict[str, object]] = []
    limited_tables = graph.tables[:max_tables]
    for table in limited_tables:
        table_summaries.append(
            {
                "qualified_name": table.qualified_name,
                "row_estimate": table.row_estimate,
                "primary_key": list(table.primary_key),
                "column_names": [column.column_name for column in table.columns],
                "outbound_edges": [
                    f"{edge.source_qualified_name}->{edge.target_qualified_name}"
                    for edge in graph.edges_from(table.table_name, schema_name=table.schema_name)
                ],
                "inbound_edges": [
                    f"{edge.source_qualified_name}->{edge.target_qualified_name}"
                    for edge in graph.edges_to(table.table_name, schema_name=table.schema_name)
                ],
            }
        )
    return {
        "table_count": len(graph.tables),
        "edge_count": len(graph.edges),
        "included_table_count": len(limited_tables),
        "truncated": len(limited_tables) != len(graph.tables),
        "tables": table_summaries,
    }


def _snapshot_to_status(snapshot: ProviderCircuitSnapshot) -> SynthesisProviderStatus:
    return SynthesisProviderStatus(
        observed_at=datetime.now(timezone.utc),
        total_requests=snapshot.total_requests,
        failures=snapshot.failures,
        error_rate=snapshot.error_rate,
        available=snapshot.available,
        cooldown_remaining_s=snapshot.cooldown_remaining_s,
    )


def _sample_output_value(field: OutputFieldContract) -> object:
    if field.type == OutputFieldType.OBJECT:
        return {child.name: _sample_output_value(child) for child in field.fields}
    if field.type == OutputFieldType.LIST:
        if field.items is None:
            return []
        return [_sample_output_value(field.items)]
    if field.type == OutputFieldType.STRING:
        return "sample"
    if field.type == OutputFieldType.INT:
        return 1
    if field.type == OutputFieldType.FLOAT:
        return 1.0
    if field.type == OutputFieldType.BOOL:
        return True
    if field.type == OutputFieldType.DATE:
        return "2026-01-01"
    if field.type == OutputFieldType.DATETIME:
        return "2026-01-01T00:00:00Z"
    if field.type == OutputFieldType.ENUM:
        return field.enum_values[0] if field.enum_values else "enum_value"
    return "sample"


def _weakened_difficulty_axes(
    *,
    previous: dict[DifficultyAxis, float] | None,
    current: dict[DifficultyAxis, float],
) -> list[str]:
    if previous is None:
        return []
    weakened: list[str] = []
    for axis, previous_value in previous.items():
        current_value = current.get(axis)
        if current_value is None or current_value < previous_value:
            weakened.append(axis.value)
    return weakened


def _strengthened_difficulty_axes(
    *,
    previous: dict[DifficultyAxis, float] | None,
    current: dict[DifficultyAxis, float],
) -> list[DifficultyAxis]:
    if previous is None:
        return []
    strengthened: list[DifficultyAxis] = []
    for axis in set(previous) | set(current):
        previous_value = previous.get(axis, 0.0)
        current_value = current.get(axis, 0.0)
        if current_value > previous_value:
            strengthened.append(axis)
    return sorted(strengthened, key=lambda axis: axis.value)


@dataclass(slots=True)
class SynthesisAgentRuntime:
    """Single-db synthesis runtime with lock-protected shared state."""

    config: AppConfig
    phase_backends: dict[SynthesisPhase, list[SynthesisStageBackend]] | None = None
    _breakers: dict[str, ProviderCircuitBreaker] = field(default_factory=dict, init=False, repr=False)
    _graph_cache: SchemaGraph | None = field(default=None, init=False, repr=False)
    _registration_pool: RegistrationSubprocessPool | None = field(
        default=None, init=False, repr=False
    )
    _atomic_tool_bundles: dict[str, AtomicToolBundle] = field(
        default_factory=dict, init=False, repr=False
    )
    _bound_db_id: str | None = field(default=None, init=False, repr=False)
    _category_failures: dict[tuple[str, CategoryTaxonomy], _CategoryFailureState] = field(
        default_factory=dict, init=False, repr=False
    )
    _bind_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _graph_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _registration_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    _atomic_tool_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _atomic_tool_materializer: AtomicToolMaterializer | None = field(
        default=None, init=False, repr=False
    )
    _category_state_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.phase_backends is None:
            from rl_task_foundry.synthesis.backend_openai_agents import (
                OpenAIAgentsSynthesisBackend,
            )

            backend = OpenAIAgentsSynthesisBackend(
                model_ref=self.config.models.composer,
                provider_config=self.config.providers[self.config.models.composer.provider],
                runtime_config=self.config.synthesis.runtime,
                session_db_path=self.config.output.traces_dir / "synthesis_sessions.sqlite",
                traces_dir=self.config.output.traces_dir / "synthesis",
            )
            self.phase_backends = {
                phase: [backend]
                for phase in (
                    SynthesisPhase.SCHEMA_EXPLORATION,
                    SynthesisPhase.CATEGORY_INFERENCE,
                    SynthesisPhase.ARTIFACT_GENERATION,
                )
            }
        self._breakers = {
            provider_name: ProviderCircuitBreaker(
                provider_name=provider_name,
                window_s=self.config.provider_resilience.circuit_breaker_window_s,
                threshold=self.config.provider_resilience.circuit_breaker_threshold,
                probe_interval_s=self.config.provider_resilience.probe_interval_s,
            )
            for provider_name in self.config.providers
        }
        self._atomic_tool_materializer = AtomicToolMaterializer.for_config(self.config)

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
        graph: SchemaGraph | None = None,
    ) -> SynthesisEnvironmentDraft:
        await self._bind_db_id(db_id)
        await self._ensure_category_available(db_id, requested_category)
        resolved_graph = graph if graph is not None else await self._introspect_graph()
        atomic_tool_bundle = await self._ensure_atomic_tool_bundle(
            db_id=db_id,
            graph=resolved_graph,
        )
        atomic_tool_set_ref = f"db://{db_id}"
        available_atomic_tools = atomic_tool_bundle.actor_tool_definitions()
        schema_summary = summarize_schema_graph(resolved_graph)
        try:
            stage_results: list[SynthesisStageResult] = []
            memory: list[SynthesisMemoryEntry] = []
            tool_traces: list[SynthesisToolTraceEntry] = []
            previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput] = {}

            for phase in (
                SynthesisPhase.SCHEMA_EXPLORATION,
                SynthesisPhase.CATEGORY_INFERENCE,
            ):
                request = SynthesisStageRequest(
                    phase=phase,
                    db_id=db_id,
                    atomic_tool_set_ref=atomic_tool_set_ref,
                    available_atomic_tools=available_atomic_tools,
                    domain_name=self.config.domain.name,
                    user_role=self.config.domain.user_role,
                    agent_role=self.config.domain.agent_role,
                    scenario_description=self.config.domain.scenario_description,
                    requested_category=requested_category,
                    attempt_index=1,
                    schema_summary=schema_summary,
                    previous_outputs=previous_outputs,
                    memory=memory[-self.config.synthesis.runtime.explicit_memory_window :],
                )
                result = await self._run_phase(request)
                stage_results.append(result)
                memory.append(result.memory_entry)
                tool_traces.extend(result.tool_traces)
                previous_outputs[phase] = result.payload

            category_payload = previous_outputs[SynthesisPhase.CATEGORY_INFERENCE]
            assert isinstance(category_payload, CategoryInferenceOutput)
            selected_category = category_payload.selected_category
            if selected_category != requested_category:
                raise SynthesisCategoryMismatchError(
                    "category inference result did not match the requested category"
                )

            (
                artifact_payload,
                registration_report,
                registration_diagnostics,
                self_consistency_diagnostics,
                instances,
                canonical_answers,
                materialized_cross_instance_set,
                self_consistency_attempts,
            ) = await self._run_artifact_generation_with_self_consistency(
                db_id=db_id,
                requested_category=requested_category,
                schema_summary=schema_summary,
                previous_outputs=previous_outputs,
                stage_results=stage_results,
                memory=memory,
                tool_traces=tool_traces,
                atomic_tool_set_ref=atomic_tool_set_ref,
                available_atomic_tools=available_atomic_tools,
            )

            materialized_at = datetime.now(timezone.utc)
            environment = self._materialize_environment(
                proposed_environment=artifact_payload.proposed_environment,
                atomic_tool_bundle=atomic_tool_bundle,
                artifacts=artifact_payload.artifacts,
                db_id=db_id,
                requested_category=requested_category,
                created_at=materialized_at,
                self_consistency_pass=self_consistency_diagnostics.passed,
                materialized_cross_instance_set=materialized_cross_instance_set,
            )
            await self._reset_category_failure_state(db_id, requested_category)

            return SynthesisEnvironmentDraft(
                created_at=materialized_at,
                db_id=db_id,
                requested_category=requested_category,
                schema_summary=schema_summary,
                selected_category=selected_category,
                environment=environment,
                atomic_tool_bundle=atomic_tool_bundle,
                artifacts=artifact_payload.artifacts,
                registration_report=registration_report,
                registration_diagnostics=registration_diagnostics,
                self_consistency_diagnostics=self_consistency_diagnostics,
                instances=instances,
                canonical_answers=canonical_answers,
                self_consistency_attempts=self_consistency_attempts,
                stage_results=stage_results,
                memory=memory,
                tool_traces=tool_traces,
                provider_status=self.provider_status(),
            )
        except SynthesisCategoryMismatchError:
            await self._record_category_discard(
                db_id,
                requested_category,
                outcome=SynthesisSelfConsistencyOutcome.CATEGORY_MISMATCH,
                error_codes=["category_mismatch"],
            )
            raise
        except SynthesisSelfConsistencyError as exc:
            last_attempt = exc.attempts[-1] if exc.attempts else None
            error_codes: list[str] = []
            if exc.last_registration_diagnostics is not None:
                error_codes.extend(exc.last_registration_diagnostics.error_codes)
            if exc.last_self_consistency_diagnostics is not None:
                error_codes.extend(exc.last_self_consistency_diagnostics.error_codes)
            await self._record_category_discard(
                db_id,
                requested_category,
                outcome=last_attempt.outcome if last_attempt is not None else None,
                error_codes=error_codes,
            )
            raise

    async def close(self) -> None:
        if self._registration_pool is not None:
            await self._registration_pool.close()
            self._registration_pool = None
        self._atomic_tool_bundles.clear()

    def provider_status(self) -> dict[str, SynthesisProviderStatus]:
        return {
            provider_name: _snapshot_to_status(breaker.snapshot())
            for provider_name, breaker in self._breakers.items()
        }

    async def category_status(
        self,
        *,
        db_id: str | None = None,
    ) -> dict[CategoryTaxonomy, SynthesisCategoryStatus]:
        resolved_db_id = db_id or self._bound_db_id
        if resolved_db_id is None:
            return {}
        if self._bound_db_id is not None and resolved_db_id != self._bound_db_id:
            raise SynthesisDbBindingError(
                "SynthesisAgentRuntime instances are single-db; create a new runtime per db_id"
            )
        now = datetime.now(timezone.utc)
        async with self._category_state_lock:
            expired = [
                key
                for key, state in self._category_failures.items()
                if key[0] == resolved_db_id
                and state.backoff_until is not None
                and state.backoff_until <= now
            ]
            for key in expired:
                del self._category_failures[key]
            statuses: dict[CategoryTaxonomy, SynthesisCategoryStatus] = {}
            for (state_db_id, category), state in self._category_failures.items():
                if state_db_id != resolved_db_id:
                    continue
                remaining = 0.0
                if state.backoff_until is not None:
                    remaining = max(0.0, (state.backoff_until - now).total_seconds())
                statuses[category] = SynthesisCategoryStatus(
                    db_id=state_db_id,
                    category=category,
                    consecutive_discards=state.consecutive_discards,
                    backed_off=state.backoff_until is not None and state.backoff_until > now,
                    backoff_until=state.backoff_until,
                    backoff_remaining_s=remaining,
                    last_outcome=state.last_outcome,
                    last_error_codes=list(state.last_error_codes),
                    last_updated_at=state.last_updated_at,
                )
            return statuses

    async def _ensure_category_available(
        self,
        db_id: str,
        category: CategoryTaxonomy,
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self._category_state_lock:
            state = self._category_failures.get((db_id, category))
            if state is None:
                return
            if state.backoff_until is None:
                return
            if state.backoff_until <= now:
                del self._category_failures[(db_id, category)]
                return
            raise SynthesisCategoryBackoffError(
                "synthesis category is temporarily backed off after repeated discards",
                db_id=db_id,
                category=category,
                consecutive_discards=state.consecutive_discards,
                backoff_until=state.backoff_until,
                last_outcome=state.last_outcome,
                last_error_codes=state.last_error_codes,
            )

    async def _record_category_discard(
        self,
        db_id: str,
        category: CategoryTaxonomy,
        *,
        outcome: SynthesisSelfConsistencyOutcome | None,
        error_codes: list[str] | None = None,
    ) -> None:
        async with self._category_state_lock:
            key = (db_id, category)
            state = self._category_failures.get(key, _CategoryFailureState())
            state.consecutive_discards += 1
            state.last_outcome = outcome
            state.last_error_codes = list(error_codes or [])
            state.last_updated_at = datetime.now(timezone.utc)
            if (
                state.consecutive_discards
                >= self.config.synthesis.runtime.max_consecutive_category_discards
            ):
                state.backoff_until = state.last_updated_at + timedelta(
                    seconds=self.config.synthesis.runtime.category_backoff_duration_s
                )
            self._category_failures[key] = state

    async def _reset_category_failure_state(
        self,
        db_id: str,
        category: CategoryTaxonomy,
    ) -> None:
        async with self._category_state_lock:
            self._category_failures.pop((db_id, category), None)

    async def _run_phase(self, request: SynthesisStageRequest) -> SynthesisStageResult:
        candidate_backends = self.phase_backends.get(request.phase, []) if self.phase_backends else []
        if not candidate_backends:
            raise SynthesisPhaseExecutionError(
                f"no synthesis backends configured for {request.phase.value}"
            )

        errors: list[str] = []
        backend_failures: list[SynthesisBackendFailure] = []
        for backend in candidate_backends:
            breaker = self._breakers[backend.provider_name]
            if not breaker.is_available():
                continue
            try:
                result = await backend.run_stage(request)
            except Exception as exc:  # pragma: no cover
                breaker.record_failure()
                errors.append(f"{backend.provider_name}/{backend.model_name}: {type(exc).__name__}")
                backend_failures.append(
                    SynthesisBackendFailure(
                        provider=backend.provider_name,
                        model=backend.model_name,
                        error_type=type(exc).__name__,
                    )
                )
                continue
            breaker.record_success()
            return result

        if errors:
            raise SynthesisPhaseExecutionError(
                f"synthesis phase {request.phase.value} failed across candidate providers: "
                + ", ".join(errors),
                phase=request.phase,
                backend_failures=backend_failures,
            )
        raise SynthesisProviderUnavailableError(
            f"all providers are in cooldown for synthesis phase {request.phase.value}",
            phase=request.phase,
        )

    async def _run_artifact_generation_with_self_consistency(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
        schema_summary: dict[str, object],
        previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput],
        stage_results: list[SynthesisStageResult],
        memory: list[SynthesisMemoryEntry],
        tool_traces: list[SynthesisToolTraceEntry],
        atomic_tool_set_ref: str,
        available_atomic_tools: list[dict[str, object]],
    ) -> tuple[
        ArtifactGenerationOutput,
        RegistrationBundleReport,
        RegistrationBundleDiagnostics,
        SynthesisSelfConsistencyDiagnostics,
        list[MaterializedInstanceRecord],
        list[MaterializedCanonicalAnswerRecord],
        CrossInstanceSet,
        list[SynthesisSelfConsistencyAttempt],
    ]:
        attempts: list[SynthesisSelfConsistencyAttempt] = []
        latest_registration_diagnostics: RegistrationBundleDiagnostics | None = None
        latest_self_consistency_diagnostics: SynthesisSelfConsistencyDiagnostics | None = None
        strongest_difficulty_vector: dict[DifficultyAxis, float] | None = None
        difficulty_crank_count = 0
        difficulty_crank_history: list[DifficultyAxis] = []
        max_iterations = self.config.synthesis.runtime.max_self_consistency_iterations

        for attempt_index in range(1, max_iterations + 1):
            request = SynthesisStageRequest(
                phase=SynthesisPhase.ARTIFACT_GENERATION,
                db_id=db_id,
                atomic_tool_set_ref=atomic_tool_set_ref,
                available_atomic_tools=available_atomic_tools,
                domain_name=self.config.domain.name,
                user_role=self.config.domain.user_role,
                agent_role=self.config.domain.agent_role,
                scenario_description=self.config.domain.scenario_description,
                requested_category=requested_category,
                attempt_index=attempt_index,
                schema_summary=schema_summary,
                previous_outputs=previous_outputs,
                memory=memory[-self.config.synthesis.runtime.explicit_memory_window :],
                latest_registration_diagnostics=latest_registration_diagnostics,
                latest_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                strongest_difficulty_vector=strongest_difficulty_vector or {},
                difficulty_crank_index=difficulty_crank_count,
                difficulty_crank_history=list(difficulty_crank_history),
            )
            result = await self._run_phase(request)
            assert isinstance(result.payload, ArtifactGenerationOutput)
            stage_results.append(result)
            memory.append(result.memory_entry)
            tool_traces.extend(result.tool_traces)

            artifact_payload = result.payload
            current_difficulty_vector = dict(
                artifact_payload.proposed_environment.task.difficulty_vector
            )
            weakened_axes = _weakened_difficulty_axes(
                previous=strongest_difficulty_vector,
                current=current_difficulty_vector,
            )
            strengthened_axes = _strengthened_difficulty_axes(
                previous=strongest_difficulty_vector,
                current=current_difficulty_vector,
            )
            if weakened_axes:
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.DIFFICULTY_WEAKENED,
                    provider=result.provider,
                    model=result.model,
                    memory_summary=result.memory_entry.summary,
                    error_message=(
                        "artifact generation reduced difficulty on axes: "
                        + ",".join(weakened_axes)
                    ),
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after difficulty weakening",
                        attempts=attempts,
                        last_registration_diagnostics=latest_registration_diagnostics,
                        last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                    )
                continue
            if len(strengthened_axes) > 1:
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.DIFFICULTY_CRANK_INVALID,
                    provider=result.provider,
                    model=result.model,
                    memory_summary=result.memory_entry.summary,
                    error_message=(
                        "difficulty crank changed multiple axes: "
                        + ",".join(axis.value for axis in strengthened_axes)
                    ),
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after invalid difficulty crank changes",
                        attempts=attempts,
                        last_registration_diagnostics=latest_registration_diagnostics,
                        last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                    )
                continue
            if len(strengthened_axes) == 1:
                next_crank_count = difficulty_crank_count + 1
                if next_crank_count > self.config.synthesis.runtime.max_difficulty_cranks:
                    attempt = SynthesisSelfConsistencyAttempt(
                        attempt_index=attempt_index,
                        outcome=SynthesisSelfConsistencyOutcome.DIFFICULTY_CRANK_LIMIT_EXCEEDED,
                        provider=result.provider,
                        model=result.model,
                        memory_summary=result.memory_entry.summary,
                        error_message=(
                            "difficulty crank limit exceeded: "
                            f"max={self.config.synthesis.runtime.max_difficulty_cranks}"
                        ),
                    )
                    attempts.append(attempt)
                    memory.append(self._attempt_feedback_entry(attempt))
                    if attempt_index >= max_iterations:
                        raise SynthesisSelfConsistencyError(
                            "artifact generation exhausted self-consistency budget after exceeding the difficulty crank limit",
                            attempts=attempts,
                            last_registration_diagnostics=latest_registration_diagnostics,
                            last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                        )
                    continue
                difficulty_crank_count = next_crank_count
                difficulty_crank_history.append(strengthened_axes[0])
            if artifact_payload.proposed_environment.task.category != requested_category:
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.CATEGORY_MISMATCH,
                    provider=result.provider,
                    model=result.model,
                    memory_summary=result.memory_entry.summary,
                    error_message="artifact generation returned a task with the wrong category",
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                latest_registration_diagnostics = None
                latest_self_consistency_diagnostics = None
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget on category mismatch",
                        attempts=attempts,
                )
                continue
            strongest_difficulty_vector = current_difficulty_vector

            try:
                registration_report, registration_diagnostics = await self._execute_registration_gate(
                    artifact_payload.artifacts,
                    proposed_environment=artifact_payload.proposed_environment,
                )
            except SynthesisRegistrationError as exc:
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.REGISTRATION_FAILED,
                    provider=result.provider,
                    model=result.model,
                    memory_summary=result.memory_entry.summary,
                    error_message=str(exc),
                    registration_diagnostics=exc.diagnostics,
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                latest_registration_diagnostics = exc.diagnostics
                latest_self_consistency_diagnostics = None
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after registration failures",
                        attempts=attempts,
                        last_registration_diagnostics=exc.diagnostics,
                    ) from exc
                continue

            self_consistency_diagnostics = await self._execute_self_consistency_check(
                bundle=artifact_payload.artifacts,
                proposed_environment=artifact_payload.proposed_environment,
            )
            if not self_consistency_diagnostics.passed:
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
                    provider=result.provider,
                    model=result.model,
                    memory_summary=result.memory_entry.summary,
                    error_message="solution output did not satisfy the primary verifier",
                    registration_diagnostics=registration_diagnostics,
                    self_consistency_diagnostics=self_consistency_diagnostics,
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                latest_registration_diagnostics = registration_diagnostics
                latest_self_consistency_diagnostics = self_consistency_diagnostics
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after verifier failures",
                        attempts=attempts,
                        last_registration_diagnostics=registration_diagnostics,
                        last_self_consistency_diagnostics=self_consistency_diagnostics,
                    )
                continue

            try:
                (
                    instances,
                    canonical_answers,
                    materialized_cross_instance_set,
                ) = self._materialize_instances_and_canonical_answers(
                    proposed_environment=artifact_payload.proposed_environment,
                    self_consistency_diagnostics=self_consistency_diagnostics,
                )
            except CanonicalizationError:
                materialization_diagnostics = self_consistency_diagnostics.model_copy(
                    update={
                        "passed": False,
                        "error_codes": [
                            *self_consistency_diagnostics.error_codes,
                            "canonical_answer_schema_mismatch",
                        ],
                    }
                )
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
                    provider=result.provider,
                    model=result.model,
                    memory_summary=result.memory_entry.summary,
                    error_message="solution output could not be canonicalized against the output schema",
                    registration_diagnostics=registration_diagnostics,
                    self_consistency_diagnostics=materialization_diagnostics,
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                latest_registration_diagnostics = registration_diagnostics
                latest_self_consistency_diagnostics = materialization_diagnostics
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after canonical answer materialization failures",
                        attempts=attempts,
                        last_registration_diagnostics=registration_diagnostics,
                        last_self_consistency_diagnostics=materialization_diagnostics,
                    )
                continue

            passed_attempt = SynthesisSelfConsistencyAttempt(
                attempt_index=attempt_index,
                outcome=SynthesisSelfConsistencyOutcome.PASSED,
                provider=result.provider,
                model=result.model,
                memory_summary=result.memory_entry.summary,
                registration_diagnostics=registration_diagnostics,
                self_consistency_diagnostics=self_consistency_diagnostics,
            )
            attempts.append(passed_attempt)
            return (
                artifact_payload,
                registration_report,
                registration_diagnostics,
                self_consistency_diagnostics,
                instances,
                canonical_answers,
                materialized_cross_instance_set,
                attempts,
            )

        raise SynthesisSelfConsistencyError(
            "artifact generation exhausted self-consistency budget",
            attempts=attempts,
            last_registration_diagnostics=latest_registration_diagnostics,
            last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
        )

    @staticmethod
    def _attempt_feedback_entry(
        attempt: SynthesisSelfConsistencyAttempt,
    ) -> SynthesisMemoryEntry:
        registration_diagnostics = attempt.registration_diagnostics
        self_consistency_diagnostics = attempt.self_consistency_diagnostics
        weak_signals = (
            registration_diagnostics.weak_signal_codes
            if registration_diagnostics is not None
            else []
        )
        error_codes = (
            registration_diagnostics.error_codes if registration_diagnostics is not None else []
        )
        detail_parts = [
            f"self_consistency_attempt={attempt.attempt_index}",
            f"outcome={attempt.outcome.value}",
        ]
        if attempt.error_message:
            detail_parts.append(f"message={attempt.error_message}")
        if error_codes:
            detail_parts.append(f"errors={','.join(error_codes)}")
        if weak_signals:
            detail_parts.append(f"weak_signals={','.join(weak_signals)}")
        if self_consistency_diagnostics is not None:
            if self_consistency_diagnostics.error_codes:
                detail_parts.append(
                    "self_consistency_errors="
                    + ",".join(self_consistency_diagnostics.error_codes)
                )
            if self_consistency_diagnostics.weak_signal_codes:
                detail_parts.append(
                    "self_consistency_weak_signals="
                    + ",".join(self_consistency_diagnostics.weak_signal_codes)
                )
            if self_consistency_diagnostics.facts_match_result is not None:
                detail_parts.append(
                    f"facts_match={self_consistency_diagnostics.facts_match_result}"
                )
            if self_consistency_diagnostics.check_constraints_result is not None:
                detail_parts.append(
                    "check_constraints="
                    f"{self_consistency_diagnostics.check_constraints_result}"
                )
            if self_consistency_diagnostics.verify_result is not None:
                detail_parts.append(f"verify={self_consistency_diagnostics.verify_result}")
        return SynthesisMemoryEntry(
            phase=SynthesisPhase.ARTIFACT_GENERATION,
            provider="runtime",
            model="self_consistency",
            summary="; ".join(detail_parts),
            turn_count=0,
        )

    async def _introspect_graph(self) -> SchemaGraph:
        if self._graph_cache is not None:
            return self._graph_cache
        async with self._graph_lock:
            if self._graph_cache is not None:
                return self._graph_cache
            introspector = PostgresSchemaIntrospector(
                database=self.config.database,
                default_visibility=self.config.privacy.default_visibility,
                visibility_overrides=self.config.privacy.visibility_overrides,
            )
            self._graph_cache = await introspector.introspect()
        return self._graph_cache

    async def _ensure_atomic_tool_bundle(
        self,
        *,
        db_id: str,
        graph: SchemaGraph,
    ) -> AtomicToolBundle:
        existing = self._atomic_tool_bundles.get(db_id)
        if existing is not None:
            return existing
        async with self._atomic_tool_lock:
            existing = self._atomic_tool_bundles.get(db_id)
            if existing is not None:
                return existing
            bundle = AtomicToolGenerator(self.config.atomic_tools).generate_bundle(
                graph,
                db_id=db_id,
            )
            assert self._atomic_tool_materializer is not None
            self._atomic_tool_materializer.materialize_bundle(bundle)
            self._atomic_tool_bundles[db_id] = bundle
            return bundle

    def _atomic_tool_bundle_for_bound_db(self) -> AtomicToolBundle:
        if self._bound_db_id is None:
            raise RuntimeError("atomic tool bundle requested before db binding")
        bundle = self._atomic_tool_bundles.get(self._bound_db_id)
        if bundle is None:
            raise RuntimeError("atomic tool bundle requested before generation")
        return bundle

    async def _execute_registration_gate(
        self,
        bundle: GeneratedArtifactBundle,
        *,
        proposed_environment: ProposedEnvironmentDraft,
    ) -> tuple[RegistrationBundleReport, RegistrationBundleDiagnostics]:
        try:
            report = await self._run_registration_gate(
                bundle=bundle,
                proposed_environment=proposed_environment,
            )
        except Exception as exc:
            raise SynthesisRegistrationError("registration gate execution failed") from exc
        diagnostics = build_registration_diagnostics(report)
        if report.status != RegistrationBundleStatus.PASSED:
            raise SynthesisRegistrationError(
                "generated artifacts failed registration validation",
                report=report,
                diagnostics=diagnostics,
            )
        return report, diagnostics

    async def _execute_self_consistency_check(
        self,
        *,
        bundle: GeneratedArtifactBundle,
        proposed_environment: ProposedEnvironmentDraft,
    ) -> SynthesisSelfConsistencyDiagnostics:
        result = await self._run_self_consistency_check(
            bundle=bundle,
            proposed_environment=proposed_environment,
        )
        return self._build_self_consistency_diagnostics(result)

    async def _run_self_consistency_check(
        self,
        *,
        bundle: GeneratedArtifactBundle,
        proposed_environment: ProposedEnvironmentDraft,
    ) -> RegistrationSelfConsistencyResult:
        if self._registration_pool is None:
            async with self._registration_lock:
                if self._registration_pool is None:
                    self._registration_pool = await RegistrationSubprocessPool.start(
                        self.config
                    )
        return await self._registration_pool.run_self_consistency_check(
            atomic_tool_set_ref=f"db://{self._bound_db_id}",
            solution_source=bundle.solution_source,
            verifier_source=bundle.verifier_source,
            shadow_verifier_source=bundle.shadow_verifier_source,
            expected_fact_keys=[
                fact.key for fact in proposed_environment.verifier.facts_schema.facts
            ],
        )

    @staticmethod
    def _build_self_consistency_diagnostics(
        result: RegistrationSelfConsistencyResult,
    ) -> SynthesisSelfConsistencyDiagnostics:
        weak_signal_codes: list[str] = []
        if result.solution_tool_calls == 0:
            weak_signal_codes.append("solution_missing_tool_usage")
        if result.verifier_tool_calls == 0:
            weak_signal_codes.append("verifier_missing_tool_usage")
        if result.shadow_verifier_tool_calls == 0:
            weak_signal_codes.append("shadow_verifier_missing_tool_usage")
        if result.fetch_facts_answer_reads == 0:
            weak_signal_codes.append("fetch_facts_missing_answer_usage_runtime")
        if result.facts_match_answer_reads == 0:
            weak_signal_codes.append("facts_match_missing_answer_usage_runtime")
        if result.facts_match_facts_reads == 0:
            weak_signal_codes.append("facts_match_missing_facts_usage_runtime")
        if result.check_constraints_facts_reads == 0:
            weak_signal_codes.append("check_constraints_missing_facts_usage_runtime")
        return SynthesisSelfConsistencyDiagnostics(
            passed=(
                not result.errors
                and bool(result.verify_result)
                and (result.shadow_verify_result is None or bool(result.shadow_verify_result))
            ),
            error_codes=[error.code for error in result.errors],
            weak_signal_codes=weak_signal_codes,
            answer=result.answer,
            solution_tool_calls=result.solution_tool_calls,
            verifier_tool_calls=result.verifier_tool_calls,
            shadow_verifier_tool_calls=result.shadow_verifier_tool_calls,
            fetch_facts_return_keys=list(result.fetch_facts_return_keys),
            expected_fact_keys=list(result.expected_fact_keys),
            missing_fact_keys=list(result.missing_fact_keys),
            extra_fact_keys=list(result.extra_fact_keys),
            fetch_facts_answer_reads=result.fetch_facts_answer_reads,
            facts_match_answer_reads=result.facts_match_answer_reads,
            facts_match_facts_reads=result.facts_match_facts_reads,
            check_constraints_facts_reads=result.check_constraints_facts_reads,
            facts_match_result=result.facts_match_result,
            check_constraints_result=result.check_constraints_result,
            verify_result=result.verify_result,
            shadow_verify_result=result.shadow_verify_result,
        )

    async def _run_registration_gate(
        self,
        *,
        bundle: GeneratedArtifactBundle,
        proposed_environment: ProposedEnvironmentDraft,
    ) -> RegistrationBundleReport:
        if self._registration_pool is None:
            async with self._registration_lock:
                if self._registration_pool is None:
                    self._registration_pool = await RegistrationSubprocessPool.start(
                        self.config
                    )
        return await run_registration_bundle(
            config=self.config,
            bundle=bundle,
            atomic_tool_set_ref=f"db://{self._bound_db_id}",
            pool=self._registration_pool,
            verifier_probe_specs=self._build_verifier_probe_specs(proposed_environment),
        )

    def _materialize_environment(
        self,
        *,
        proposed_environment: ProposedEnvironmentDraft,
        atomic_tool_bundle: AtomicToolBundle,
        artifacts: GeneratedArtifactBundle,
        db_id: str,
        requested_category: CategoryTaxonomy,
        created_at: datetime,
        self_consistency_pass: bool,
        materialized_cross_instance_set: CrossInstanceSet,
    ) -> EnvironmentContract:
        task_payload = proposed_environment.task.model_dump(mode="python")
        task_signature = self._signature_for_payload(task_payload)
        tool_signature = self._signature_for_text(atomic_tool_bundle.source)
        verifier_signature = self._signature_for_text(
            artifacts.verifier_source + "\n---shadow---\n" + artifacts.shadow_verifier_source
        )
        env_id = self._build_env_id(
            db_id=db_id,
            category=requested_category,
            task_signature=task_signature,
            tool_signature=tool_signature,
            verifier_signature=verifier_signature,
        )
        payload = proposed_environment.model_dump(mode="python")
        payload.update(
            {
                "env_id": env_id,
                "db_id": db_id,
                "domain": self.config.domain.name,
                "category": requested_category,
                "atomic_tool_set_ref": f"db://{db_id}",
                "difficulty_vector": proposed_environment.task.difficulty_vector,
                "created_at": created_at,
                "generator_version": CURRENT_SYNTHESIS_GENERATOR_VERSION,
                "tool_signature": tool_signature,
                "task_signature": task_signature,
                "verifier_signature": verifier_signature,
                "status": EnvironmentStatus.DRAFT,
                "quality_metrics": EnvironmentQualityMetrics(
                    self_consistency_pass=self_consistency_pass
                ).model_dump(mode="python"),
                "rollout_constraints": self._build_rollout_constraints().model_dump(
                    mode="python"
                ),
                "cross_instance_set": materialized_cross_instance_set.model_dump(mode="python"),
                "task": proposed_environment.task.model_copy(
                    update={"category": requested_category}
                ).model_dump(mode="python"),
            }
        )
        return EnvironmentContract.model_validate(payload)

    def _build_rollout_constraints(self) -> RolloutConstraintsContract:
        return RolloutConstraintsContract(
            max_turns=self.config.solver_runtime.max_turns,
            max_episode_duration_ms=(
                self.config.database.statement_timeout_ms * self.config.solver_runtime.max_turns
            ),
            max_tool_rows=self.config.atomic_tools.bounded_result_limit,
        )

    def _materialize_instances_and_canonical_answers(
        self,
        *,
        proposed_environment: ProposedEnvironmentDraft,
        self_consistency_diagnostics: SynthesisSelfConsistencyDiagnostics,
    ) -> tuple[
        list[MaterializedInstanceRecord],
        list[MaterializedCanonicalAnswerRecord],
        CrossInstanceSet,
    ]:
        canonical_input = self._normalize_solution_answer(
            proposed_environment.task.output_schema,
            self_consistency_diagnostics.answer,
        )
        canonical_answer = canonicalize_output(
            proposed_environment.task.output_schema,
            canonical_input,
        )
        canonical_answer_json = canonical_json(canonical_answer)
        solution_fingerprint = self._signature_for_text(canonical_answer_json)
        instance_id = "instance_0001"
        params = dict(proposed_environment.task.instance_parameters)
        instance = MaterializedInstanceRecord(
            instance_id=instance_id,
            rendered_user_prompt=build_rendered_user_prompt(proposed_environment.task),
            params=params,
            anchor_values={},
        )
        canonical_record = MaterializedCanonicalAnswerRecord(
            instance_id=instance_id,
            canonical_answer=canonical_answer,
            canonical_answer_json=canonical_answer_json,
            solution_fingerprint=solution_fingerprint,
        )
        cross_instance_set = CrossInstanceSet(
            # C8 materializes a single bootstrap instance. Keep the persisted
            # cross-instance contract aligned with the records we actually emit.
            minimum_required=1,
            require_distinct_solution_fingerprints=(
                proposed_environment.cross_instance_set.require_distinct_solution_fingerprints
            ),
            instances=[
                InstanceContract(
                    instance_id=instance_id,
                    anchor_values={},
                    parameter_values=params,
                    expected_solution_fingerprint=solution_fingerprint,
                )
            ],
        )
        return [instance], [canonical_record], cross_instance_set

    @staticmethod
    def _normalize_solution_answer(
        output_schema: OutputSchemaContract,
        answer: object,
    ) -> object:
        root = output_schema.root
        if (
            isinstance(answer, dict)
            and set(answer) == {root.name}
            and (
                root.type is not OutputFieldType.OBJECT
                or root.name not in {child.name for child in root.fields}
            )
        ):
            return answer[root.name]
        return answer

    def _build_verifier_probe_specs(
        self,
        proposed_environment: ProposedEnvironmentDraft,
    ) -> dict[RegistrationArtifactName, VerifierProbeSpec]:
        answer_sample = _sample_output_value(proposed_environment.task.output_schema.root)
        return {
            RegistrationArtifactName.VERIFIER: VerifierProbeSpec(
                answer_sample=answer_sample,
                facts_schema=proposed_environment.verifier.facts_schema,
            ),
            RegistrationArtifactName.SHADOW_VERIFIER: VerifierProbeSpec(
                answer_sample=answer_sample,
                facts_schema=proposed_environment.shadow_verifier.facts_schema,
            ),
        }

    async def _bind_db_id(self, db_id: str) -> None:
        async with self._bind_lock:
            if self._bound_db_id is None:
                self._bound_db_id = db_id
                return
            if self._bound_db_id != db_id:
                raise SynthesisDbBindingError(
                    "SynthesisAgentRuntime instances are single-db; create a new runtime per db_id"
                )

    @staticmethod
    def _signature_for_text(source: str) -> str:
        return f"sha256:{sha256(source.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _signature_for_payload(payload: dict[str, object]) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return f"sha256:{sha256(normalized.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _build_env_id(
        *,
        db_id: str,
        category: CategoryTaxonomy,
        task_signature: str,
        tool_signature: str,
        verifier_signature: str,
    ) -> str:
        digest = sha256(
            f"{db_id}|{category.value}|{task_signature}|{tool_signature}|{verifier_signature}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return f"env_{category.value}_{digest}"
