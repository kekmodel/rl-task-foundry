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
from rl_task_foundry.infra.db import DatabasePools
from rl_task_foundry.pipeline.provider_resilience import (
    ProviderCircuitBreaker,
    ProviderCircuitSnapshot,
)
from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    ConstraintSummaryItem,
    CrossInstanceSet,
    DIFFICULTY_CRANK_ORDER,
    DifficultyAxis,
    DifficultyVectorContract,
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnvironmentStatus,
    InstanceContract,
    InstanceSpaceContract,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    ShadowVerifierContract,
    SolutionContract,
    StrictModel,
    TaskContract,
    VerifierContract,
    difficulty_vector_json,
    flatten_difficulty_vector,
)
from rl_task_foundry.synthesis.canonicalize import (
    CanonicalizationError,
    canonical_json,
    canonicalize_output,
)
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle, AtomicToolGenerator
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.phase_monitor import (
    PipelinePhaseMonitorLogger,
    default_phase_monitor_log_path,
)
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
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
from rl_task_foundry.synthesis.tool_runtime import (
    ToolExecutor,
    bind_atomic_tool_executor,
    load_atomic_tool_module,
)

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
    LABEL_CONSTRUCTION = "label_construction"
    TASK_SYNTHESIS = "task_synthesis"
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
    sample_observations: list[str] = Field(default_factory=list, min_length=1)
    memory_summary: str = "schema exploration completed"


class CategoryInferenceOutput(StrictModel):
    selected_category: CategoryTaxonomy
    rationale: str
    memory_summary: str = "category inference completed"


class LabelConstructionOutput(StrictModel):
    canonical_answer_json: str = Field(min_length=1)
    output_schema: OutputSchemaContract
    difficulty_vector: DifficultyVectorContract = Field(default_factory=DifficultyVectorContract)
    instance_parameters: dict[str, object] = Field(default_factory=dict)
    label_summary: str
    memory_summary: str = "label construction completed"


class TaskSynthesisOutput(StrictModel):
    question: str
    constraint_summary: list[ConstraintSummaryItem] = Field(default_factory=list)
    instance_space: InstanceSpaceContract
    memory_summary: str = "task synthesis completed"


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
    SchemaExplorationOutput
    | CategoryInferenceOutput
    | LabelConstructionOutput
    | TaskSynthesisOutput
    | ArtifactGenerationOutput
)


class SynthesisStageRequest(StrictModel):
    phase: SynthesisPhase
    db_id: str
    atomic_tool_set_ref: str | None = None
    available_atomic_tools: list[dict[str, object]] = Field(default_factory=list)
    domain_name: str
    task_language: str = "ko"
    user_role: str | None = None
    agent_role: str | None = None
    scenario_description: str
    requested_category: CategoryTaxonomy | None = None
    attempt_index: int = 1
    schema_summary: dict[str, object] = Field(default_factory=dict)
    previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput] = Field(default_factory=dict)
    memory: list[SynthesisMemoryEntry] = Field(default_factory=list)
    latest_registration_diagnostics: RegistrationBundleDiagnostics | None = None
    latest_self_consistency_diagnostics: SynthesisSelfConsistencyDiagnostics | None = None
    strongest_difficulty_vector: DifficultyVectorContract = Field(
        default_factory=DifficultyVectorContract
    )
    difficulty_crank_index: int = Field(default=0, ge=0)
    difficulty_crank_history: list[DifficultyAxis] = Field(default_factory=list)
    next_crank_axis: DifficultyAxis | None = None
    current_diversity: int | None = Field(default=None, ge=0)
    max_diversity: int | None = Field(default=None, ge=0)
    current_scale: int | None = Field(default=None, ge=0)
    crank_hint: str | None = None
    latest_quality_gate_feedback: SynthesisQualityGateFeedback | None = None


class SynthesisStageResult(StrictModel):
    phase: SynthesisPhase
    provider: str
    model: str
    payload: SynthesisPhaseOutput
    payload_repair_codes: list[str] = Field(default_factory=list)
    memory_entry: SynthesisMemoryEntry
    tool_traces: list[SynthesisToolTraceEntry] = Field(default_factory=list)


class SynthesisQualityGateFeedback(StrictModel):
    status: str
    pass_rate: float = Field(ge=0.0, le=1.0)
    ci_lower: float = Field(ge=0.0, le=1.0)
    ci_upper: float = Field(ge=0.0, le=1.0)
    matched_solver_runs: int = Field(ge=0)
    total_solver_runs: int = Field(ge=0)
    previous_env_id: str | None = None
    previous_question: str | None = None
    previous_rendered_user_prompt: str | None = None
    previous_semantic_dedup_text: str | None = None
    previous_difficulty_vector: DifficultyVectorContract | None = None
    previous_canonical_answers: list[str] = Field(default_factory=list)
    previous_solution_fingerprints: list[str] = Field(default_factory=list)


class SynthesisDifficultyRetrySeed(StrictModel):
    strongest_difficulty_vector: DifficultyVectorContract = Field(
        default_factory=DifficultyVectorContract
    )
    difficulty_crank_index: int = Field(default=0, ge=0)
    difficulty_crank_history: list[DifficultyAxis] = Field(default_factory=list)
    retry_requires_harder: bool = False
    latest_quality_gate_feedback: SynthesisQualityGateFeedback | None = None

    def requested_axis(self) -> DifficultyAxis:
        return _next_difficulty_crank_axis(history=self.difficulty_crank_history)

    def consume_requested_crank(self) -> SynthesisDifficultyRetrySeed:
        requested_axis = self.requested_axis()
        return self.model_copy(
            update={
                "difficulty_crank_index": self.difficulty_crank_index + 1,
                "difficulty_crank_history": [*self.difficulty_crank_history, requested_axis],
            }
        )


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
    payload_repair_codes: list[str] = Field(default_factory=list)
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
    previous: DifficultyVectorContract | None,
    current: DifficultyVectorContract,
) -> list[str]:
    if previous is None:
        return []
    weakened: list[str] = []
    previous_flat = flatten_difficulty_vector(previous)
    current_flat = flatten_difficulty_vector(current)
    for axis, previous_value in previous_flat.items():
        current_value = current_flat.get(axis, 0.0)
        if current_value is None or current_value < previous_value:
            weakened.append(axis.value)
    return weakened


def _strengthened_difficulty_axes(
    *,
    previous: DifficultyVectorContract | None,
    current: DifficultyVectorContract,
) -> list[DifficultyAxis]:
    if previous is None:
        return []
    previous_flat = flatten_difficulty_vector(previous)
    current_flat = flatten_difficulty_vector(current)
    strengthened: list[DifficultyAxis] = []
    for axis in set(previous_flat) | set(current_flat):
        previous_value = previous_flat.get(axis, 0.0)
        current_value = current_flat.get(axis, 0.0)
        if current_value > previous_value:
            strengthened.append(axis)
    return sorted(
        strengthened,
        key=lambda axis: DIFFICULTY_CRANK_ORDER.index(axis),
    )


def _is_valid_difficulty_crank_step(
    *,
    history: list[DifficultyAxis],
    strengthened_axes: list[DifficultyAxis],
) -> bool:
    if len(strengthened_axes) != 1:
        return False
    if not history:
        return True
    last_axis = history[-1]
    last_index = DIFFICULTY_CRANK_ORDER.index(last_axis)
    allowed = {last_axis}
    if last_index + 1 < len(DIFFICULTY_CRANK_ORDER):
        allowed.add(DIFFICULTY_CRANK_ORDER[last_index + 1])
    return strengthened_axes[0] in allowed


def _output_slot_count(field: OutputFieldContract) -> int:
    if field.type == OutputFieldType.OBJECT:
        return sum(_output_slot_count(child) for child in field.fields) or 1
    if field.type == OutputFieldType.LIST and field.items is not None:
        return _output_slot_count(field.items)
    return 1


def _search_cost_diversity(
    *,
    graph: SchemaGraph,
    bundle: GeneratedArtifactBundle | None,
    available_atomic_tools: list[dict[str, object]],
) -> int:
    if bundle is None:
        return 0
    tool_names = [
        str(tool["name"])
        for tool in available_atomic_tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ]
    source = "\n".join(
        [
            bundle.solution_source,
            bundle.verifier_source,
            bundle.shadow_verifier_source,
        ]
    )
    referenced_tool_names = [name for name in tool_names if f"tools.{name}(" in source]
    referenced_tables: set[str] = set()
    for name in referenced_tool_names:
        for table in graph.tables:
            table_slug = table.table_name.lower()
            if table_slug in name:
                referenced_tables.add(table.qualified_name)
    if referenced_tables:
        return len(referenced_tables)
    return len(referenced_tool_names)


def _search_cost_max_diversity(graph: SchemaGraph) -> int:
    return max(1, len(graph.tables))


def _search_cost_scale(
    *,
    bundle: GeneratedArtifactBundle | None,
    available_atomic_tools: list[dict[str, object]],
) -> int:
    if bundle is None:
        return 0
    tool_names = [
        str(tool["name"])
        for tool in available_atomic_tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ]
    source = "\n".join(
        [
            bundle.solution_source,
            bundle.verifier_source,
            bundle.shadow_verifier_source,
        ]
    )
    total_calls = 0
    for name in tool_names:
        total_calls += source.count(f"tools.{name}(")
    return total_calls


def _solution_space_diversity(task: TaskContract | None) -> int:
    if task is None:
        return 0
    return _output_slot_count(task.output_schema.root)


def _solution_space_max_diversity(graph: SchemaGraph) -> int:
    return max(1, len(graph.tables))


def _solution_space_scale(task: TaskContract | None) -> int:
    if task is None:
        return 0
    return max(1, len(task.instance_parameters) or _output_slot_count(task.output_schema.root))


def _constraint_density_diversity(task: TaskContract | None) -> int:
    if task is None:
        return 0
    return len({item.kind for item in task.constraint_summary})


def _constraint_density_max_diversity() -> int:
    return 3


def _constraint_density_scale(task: TaskContract | None) -> int:
    if task is None:
        return 0
    return len(task.constraint_summary)


def _top_level_answer_field_names(task: TaskContract) -> set[str]:
    root = task.output_schema.root
    if root.type is not OutputFieldType.OBJECT:
        return set()
    return {field.name for field in root.fields}


def _project_retry_difficulty_vector(
    *,
    previous: DifficultyVectorContract | None,
    current: DifficultyVectorContract,
    requested_axis: DifficultyAxis | None,
) -> DifficultyVectorContract | None:
    if previous is None or requested_axis is None:
        return None
    strengthened_axes = _strengthened_difficulty_axes(previous=previous, current=current)
    if requested_axis not in strengthened_axes or len(strengthened_axes) <= 1:
        return None
    projected_payload = difficulty_vector_json(previous)
    projected_payload[requested_axis.value] = float(
        max(
            getattr(previous, requested_axis.value),
            getattr(current, requested_axis.value),
        )
    )
    return DifficultyVectorContract.model_validate(projected_payload)


def _next_difficulty_crank_axis(*, history: list[DifficultyAxis]) -> DifficultyAxis:
    if not history:
        return DifficultyAxis.SEARCH_COST
    last_axis = history[-1]
    repeat_count = history.count(last_axis)
    if repeat_count < 2:
        return last_axis
    last_index = DIFFICULTY_CRANK_ORDER.index(last_axis)
    if last_index + 1 < len(DIFFICULTY_CRANK_ORDER):
        return DIFFICULTY_CRANK_ORDER[last_index + 1]
    return last_axis


def _consume_difficulty_crank_attempt(
    *,
    count: int,
    history: list[DifficultyAxis],
) -> tuple[int, list[DifficultyAxis]]:
    requested_axis = _next_difficulty_crank_axis(history=history)
    return count + 1, [*history, requested_axis]


def _difficulty_guidance(
    *,
    graph: SchemaGraph,
    task: TaskContract | None,
    bundle: GeneratedArtifactBundle | None,
    available_atomic_tools: list[dict[str, object]],
    history: list[DifficultyAxis],
) -> tuple[DifficultyAxis, int, int, int, str]:
    axis = _next_difficulty_crank_axis(history=history)
    if axis is DifficultyAxis.SEARCH_COST:
        current_diversity = _search_cost_diversity(
            graph=graph,
            bundle=bundle,
            available_atomic_tools=available_atomic_tools,
        )
        max_diversity = _search_cost_max_diversity(graph)
        current_scale = _search_cost_scale(
            bundle=bundle,
            available_atomic_tools=available_atomic_tools,
        )
        if current_diversity < max_diversity:
            hint = (
                "Increase search_cost by adding a new table join or traversal step that "
                "forces a longer evidence chain."
            )
        else:
            hint = (
                "Search-cost diversity is saturated. Increase scale by widening the candidate "
                "fanout or comparator set so the chain must inspect more evidence."
            )
        return axis, current_diversity, max_diversity, current_scale, hint
    if axis is DifficultyAxis.SOLUTION_SPACE:
        current_diversity = _solution_space_diversity(task)
        max_diversity = _solution_space_max_diversity(graph)
        current_scale = _solution_space_scale(task)
        if current_diversity < max_diversity:
            hint = (
                "Increase solution_space by adding a new output slot or restructuring the "
                "answer so the valid combination has more moving parts."
            )
        else:
            hint = (
                "Solution-space diversity is saturated. Increase scale by broadening the "
                "candidate combination width for each answer slot."
            )
        return axis, current_diversity, max_diversity, current_scale, hint
    current_diversity = _constraint_density_diversity(task)
    max_diversity = _constraint_density_max_diversity()
    current_scale = _constraint_density_scale(task)
    if current_diversity < max_diversity:
        hint = (
            "Increase constraint_density by adding a new interacting constraint type or "
            "branching condition."
        )
    else:
        hint = (
            "Constraint-density diversity is saturated. Increase scale by making existing "
            "constraints interact more tightly so valid answers become rarer."
        )
    return axis, current_diversity, max_diversity, current_scale, hint


def merge_strongest_difficulty_vector(
    previous: DifficultyVectorContract | None,
    current: DifficultyVectorContract,
) -> DifficultyVectorContract:
    if previous is None:
        return current
    return DifficultyVectorContract(
        search_cost=max(previous.search_cost, current.search_cost),
        solution_space=max(previous.solution_space, current.solution_space),
        constraint_density=max(previous.constraint_density, current.constraint_density),
    )


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
    _tool_executor_cache: dict[str, dict[str, ToolExecutor]] = field(
        default_factory=dict, init=False, repr=False
    )
    _database_pools: DatabasePools | None = field(default=None, init=False, repr=False)
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
    phase_monitor: PipelinePhaseMonitorLogger | None = None

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
                    SynthesisPhase.LABEL_CONSTRUCTION,
                    SynthesisPhase.TASK_SYNTHESIS,
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
        if self.phase_monitor is None:
            self.phase_monitor = PipelinePhaseMonitorLogger(
                phase_monitor_log_path=default_phase_monitor_log_path(
                    self.config.output.traces_dir
                ),
                flow_kind="synthesis_runtime",
                flow_id=build_flow_id("synthesis_runtime"),
            )

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
        graph: SchemaGraph | None = None,
        retry_seed: SynthesisDifficultyRetrySeed | None = None,
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
        await self._prime_phase_backends_with_atomic_tools(
            db_id=db_id,
            bundle=atomic_tool_bundle,
        )
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
                    scenario_description=self.config.domain.scenario_description,
                    requested_category=requested_category,
                    attempt_index=1,
                    schema_summary=schema_summary,
                    previous_outputs=previous_outputs,
                    memory=memory[-self.config.synthesis.runtime.explicit_memory_window :],
                )
                result = await self._run_phase(request)
                if phase is SynthesisPhase.SCHEMA_EXPLORATION:
                    self._ensure_grounded_schema_exploration(request, result)
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
                proposed_environment,
                artifacts,
                registration_report,
                registration_diagnostics,
                self_consistency_diagnostics,
                instances,
                canonical_answers,
                materialized_cross_instance_set,
                self_consistency_attempts,
            ) = await self._run_label_first_generation_with_self_consistency(
                db_id=db_id,
                requested_category=requested_category,
                graph=resolved_graph,
                schema_summary=schema_summary,
                previous_outputs=previous_outputs,
                stage_results=stage_results,
                memory=memory,
                tool_traces=tool_traces,
                atomic_tool_set_ref=atomic_tool_set_ref,
                available_atomic_tools=available_atomic_tools,
                retry_seed=retry_seed,
            )

            materialized_at = datetime.now(timezone.utc)
            environment = self._materialize_environment(
                proposed_environment=proposed_environment,
                atomic_tool_bundle=atomic_tool_bundle,
                artifacts=artifacts,
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
                artifacts=artifacts,
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
        if self._database_pools is not None:
            await self._database_pools.close()
            self._database_pools = None
        self._atomic_tool_bundles.clear()
        self._tool_executor_cache.clear()

    def _emit_phase_monitor(
        self,
        *,
        phase: str,
        status: str,
        expected_contract: dict[str, object] | None = None,
        actual_data: dict[str, object] | None = None,
        checks: dict[str, object] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        if self.phase_monitor is None:
            return
        self.phase_monitor.emit(
            phase=phase,
            status=status,
            expected_contract=expected_contract,
            actual_data=actual_data,
            checks=checks,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _artifact_present_fields(bundle: GeneratedArtifactBundle) -> list[str]:
        payload = bundle.model_dump(mode="python")
        return sorted(
            key
            for key, value in payload.items()
            if isinstance(value, str) and value.strip()
        )

    @staticmethod
    def _fact_keys(contract: VerifierContract | ShadowVerifierContract) -> list[str]:
        return [fact.key for fact in contract.facts_schema.facts]

    def _expected_contract_for_request(
        self,
        request: SynthesisStageRequest,
    ) -> dict[str, object]:
        base = {
            "db_id": request.db_id,
            "phase": request.phase.value,
            "attempt_index": request.attempt_index,
            "requested_category": (
                request.requested_category.value
                if request.requested_category is not None
                else None
            ),
            "atomic_tool_count": len(request.available_atomic_tools),
            "memory_window_size": len(request.memory),
        }
        if request.phase == SynthesisPhase.SCHEMA_EXPLORATION:
            base.update(
                {
                    "required_fields": [
                        "domain_hypothesis",
                        "candidate_categories",
                        "sample_observations",
                        "memory_summary",
                    ],
                    "candidate_categories_min_length": 1,
                    "sample_observations_min_length": 1,
                    "must_use_live_atomic_tools": True,
                }
            )
            return base
        if request.phase == SynthesisPhase.CATEGORY_INFERENCE:
            base.update(
                {
                    "required_fields": [
                        "selected_category",
                        "rationale",
                        "memory_summary",
                    ],
                    "must_match_requested_category": request.requested_category is not None,
                }
            )
            return base
        if request.phase == SynthesisPhase.LABEL_CONSTRUCTION:
            base.update(
                {
                    "required_fields": [
                        "canonical_answer_json",
                        "output_schema",
                        "difficulty_vector",
                        "label_summary",
                        "memory_summary",
                    ],
                    "label_first": True,
                }
            )
            return base
        if request.phase == SynthesisPhase.TASK_SYNTHESIS:
            base.update(
                {
                    "required_fields": [
                        "question",
                        "constraint_summary",
                        "instance_space",
                        "memory_summary",
                    ],
                    "must_surface_label_semantics": True,
                }
            )
            return base
        base.update(
            {
                "required_fields": [
                    "proposed_environment",
                    "artifacts",
                    "memory_summary",
                ],
                "required_proposed_environment_sections": [
                    "task",
                    "solution",
                    "verifier",
                    "shadow_verifier",
                    "instance_space",
                ],
                "required_artifact_fields": [
                    "solution_source",
                    "verifier_source",
                    "shadow_verifier_source",
                ],
                "must_match_requested_category": True,
                "difficulty_crank_index": request.difficulty_crank_index,
                "difficulty_crank_history": [axis.value for axis in request.difficulty_crank_history],
                "strongest_difficulty_vector": difficulty_vector_json(
                    request.strongest_difficulty_vector
                ),
                "next_crank_axis": (
                    request.next_crank_axis.value if request.next_crank_axis is not None else None
                ),
                "current_diversity": request.current_diversity,
                "max_diversity": request.max_diversity,
                "current_scale": request.current_scale,
                "crank_hint": request.crank_hint,
                "latest_quality_gate_feedback": (
                    request.latest_quality_gate_feedback.model_dump(mode="json")
                    if request.latest_quality_gate_feedback is not None
                    else None
                ),
            }
        )
        return base

    def _actual_data_for_stage_result(
        self,
        result: SynthesisStageResult,
    ) -> dict[str, object]:
        payload = result.payload
        if isinstance(payload, SchemaExplorationOutput):
            return {
                "domain_hypothesis": payload.domain_hypothesis,
                "candidate_categories": [category.value for category in payload.candidate_categories],
                "sample_observations": list(payload.sample_observations),
                "memory_summary": payload.memory_summary,
            }
        if isinstance(payload, CategoryInferenceOutput):
            return {
                "selected_category": payload.selected_category.value,
                "rationale": payload.rationale,
                "memory_summary": payload.memory_summary,
            }
        if isinstance(payload, LabelConstructionOutput):
            return {
                "canonical_answer_json": payload.canonical_answer_json,
                "difficulty_vector": difficulty_vector_json(payload.difficulty_vector),
                "instance_parameter_keys": sorted(payload.instance_parameters.keys()),
                "label_summary": payload.label_summary,
                "output_schema_root_type": payload.output_schema.root.type.value,
                "memory_summary": payload.memory_summary,
            }
        if isinstance(payload, TaskSynthesisOutput):
            return {
                "question": payload.question,
                "constraint_count": len(payload.constraint_summary),
                "instance_space_anchor_sql": payload.instance_space.anchor_query.sql,
                "instance_space_anchor_outputs": list(payload.instance_space.anchor_query.outputs),
                "memory_summary": payload.memory_summary,
            }
        proposed_environment = payload.proposed_environment
        task = proposed_environment.task
        return {
            "task_category": task.category.value,
            "question": task.question,
            "difficulty_vector": difficulty_vector_json(task.difficulty_vector),
            "user_prompt_preview": build_rendered_user_prompt(task),
            "output_schema_root_type": task.output_schema.root.type.value,
            "constraint_count": len(task.constraint_summary),
            "instance_parameter_keys": sorted(task.instance_parameters.keys()),
            "verifier_fact_keys": self._fact_keys(proposed_environment.verifier),
            "shadow_verifier_fact_keys": self._fact_keys(
                proposed_environment.shadow_verifier
            ),
            "artifact_fields_present": self._artifact_present_fields(payload.artifacts),
            "memory_summary": payload.memory_summary,
        }

    def _checks_for_stage_result(
        self,
        request: SynthesisStageRequest,
        result: SynthesisStageResult,
    ) -> dict[str, object]:
        payload = result.payload
        if isinstance(payload, SchemaExplorationOutput):
            return {
                "domain_hypothesis_present": bool(payload.domain_hypothesis.strip()),
                "candidate_categories_non_empty": len(payload.candidate_categories) > 0,
                "sample_observations_non_empty": len(payload.sample_observations) > 0,
                "tool_traces_present": len(result.tool_traces) > 0,
                "requested_category_in_candidates": (
                    request.requested_category.value
                    in [category.value for category in payload.candidate_categories]
                    if request.requested_category is not None
                    else None
                ),
            }
        if isinstance(payload, CategoryInferenceOutput):
            return {
                "selected_category_present": bool(payload.selected_category.value),
                "selected_category_matches_requested": (
                    payload.selected_category == request.requested_category
                    if request.requested_category is not None
                    else None
                ),
                "rationale_present": bool(payload.rationale.strip()),
            }
        if isinstance(payload, LabelConstructionOutput):
            return {
                "canonical_answer_json_present": bool(payload.canonical_answer_json.strip()),
                "label_summary_present": bool(payload.label_summary.strip()),
                "difficulty_vector_present": payload.difficulty_vector.total_score() >= 0.0,
            }
        if isinstance(payload, TaskSynthesisOutput):
            return {
                "question_present": bool(payload.question.strip()),
                "anchor_outputs_present": bool(payload.instance_space.anchor_query.outputs),
                "constraint_summary_present": len(payload.constraint_summary) > 0,
            }
        proposed_environment = payload.proposed_environment
        return {
            "task_category_matches_requested": (
                proposed_environment.task.category == request.requested_category
                if request.requested_category is not None
                else None
            ),
            "solution_source_present": bool(payload.artifacts.solution_source.strip()),
            "verifier_source_present": bool(payload.artifacts.verifier_source.strip()),
            "shadow_verifier_source_present": bool(
                payload.artifacts.shadow_verifier_source.strip()
            ),
            "verifier_fact_keys_match_shadow": self._fact_keys(
                proposed_environment.verifier
            )
            == self._fact_keys(proposed_environment.shadow_verifier),
            "difficulty_vector_present": bool(
                proposed_environment.task.difficulty_vector.nonzero_axes()
            ),
        }

    def _log_stage_result(
        self,
        request: SynthesisStageRequest,
        result: SynthesisStageResult,
    ) -> None:
        self._emit_phase_monitor(
            phase=request.phase.value,
            status="completed",
            expected_contract=self._expected_contract_for_request(request),
            actual_data=self._actual_data_for_stage_result(result),
            checks=self._checks_for_stage_result(request, result),
            diagnostics={
                "provider": result.provider,
                "model": result.model,
                "payload_repair_codes": list(result.payload_repair_codes),
                "tool_trace_count": len(result.tool_traces),
                "turn_count": result.memory_entry.turn_count,
                "token_usage": dict(result.memory_entry.token_usage),
                "transcript_ref": result.memory_entry.transcript_ref,
                "tool_trace_ref": result.memory_entry.tool_trace_ref,
            },
        )

    def _log_stage_failure(
        self,
        request: SynthesisStageRequest,
        *,
        status: str,
        error_message: str,
        backend_failures: list[SynthesisBackendFailure] | None = None,
    ) -> None:
        self._emit_phase_monitor(
            phase=request.phase.value,
            status=status,
            expected_contract=self._expected_contract_for_request(request),
            actual_data={},
            checks={},
            diagnostics={
                "error_message": error_message,
                "backend_failures": [
                    {
                        "provider": failure.provider,
                        "model": failure.model,
                        "error_type": failure.error_type,
                    }
                    for failure in (backend_failures or [])
                ],
            },
        )

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
            self._log_stage_result(request, result)
            return result

        if errors:
            self._log_stage_failure(
                request,
                status="failed",
                error_message=(
                    f"synthesis phase {request.phase.value} failed across candidate providers: "
                    + ", ".join(errors)
                ),
                backend_failures=backend_failures,
            )
            raise SynthesisPhaseExecutionError(
                f"synthesis phase {request.phase.value} failed across candidate providers: "
                + ", ".join(errors),
                phase=request.phase,
                backend_failures=backend_failures,
            )
        self._log_stage_failure(
            request,
            status="unavailable",
            error_message=(
                f"all providers are in cooldown for synthesis phase {request.phase.value}"
            ),
        )
        raise SynthesisProviderUnavailableError(
            f"all providers are in cooldown for synthesis phase {request.phase.value}",
            phase=request.phase,
        )

    async def _run_label_first_generation_with_self_consistency(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
        graph: SchemaGraph,
        schema_summary: dict[str, object],
        previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput],
        stage_results: list[SynthesisStageResult],
        memory: list[SynthesisMemoryEntry],
        tool_traces: list[SynthesisToolTraceEntry],
        atomic_tool_set_ref: str,
        available_atomic_tools: list[dict[str, object]],
        retry_seed: SynthesisDifficultyRetrySeed | None = None,
    ) -> tuple[
        ProposedEnvironmentDraft,
        GeneratedArtifactBundle,
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
        latest_artifact_payload: ArtifactGenerationOutput | None = None
        latest_authoritative_task: TaskContract | None = None
        base_previous_outputs = dict(previous_outputs)
        strongest_difficulty_vector: DifficultyVectorContract | None = (
            retry_seed.strongest_difficulty_vector if retry_seed is not None else None
        )
        difficulty_crank_count = (
            retry_seed.difficulty_crank_index if retry_seed is not None else 0
        )
        difficulty_crank_history: list[DifficultyAxis] = list(
            retry_seed.difficulty_crank_history if retry_seed is not None else []
        )
        retry_requires_harder = (
            retry_seed.retry_requires_harder if retry_seed is not None else False
        )
        max_iterations = self.config.synthesis.runtime.max_self_consistency_iterations

        for attempt_index in range(1, max_iterations + 1):
            if (
                retry_requires_harder
                and difficulty_crank_count >= self.config.synthesis.runtime.max_difficulty_cranks
            ):
                next_limit_axis = _next_difficulty_crank_axis(history=difficulty_crank_history)
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="difficulty_crank_limit_exceeded",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "max_difficulty_cranks": (
                            self.config.synthesis.runtime.max_difficulty_cranks
                        ),
                    },
                    actual_data={
                        "difficulty_crank_index": difficulty_crank_count,
                        "difficulty_crank_history": [axis.value for axis in difficulty_crank_history],
                        "next_crank_axis": next_limit_axis.value,
                    },
                    checks={"retry_requires_harder": True},
                    diagnostics={},
                )
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.DIFFICULTY_CRANK_LIMIT_EXCEEDED,
                    provider="runtime",
                    model="runtime",
                    memory_summary="difficulty crank budget exhausted before next retry",
                    error_message=(
                        "difficulty crank limit exceeded: "
                        f"max={self.config.synthesis.runtime.max_difficulty_cranks}"
                    ),
                )
                attempts.append(attempt)
                raise SynthesisSelfConsistencyError(
                    "label-first synthesis exhausted self-consistency budget after exceeding the difficulty crank limit",
                    attempts=attempts,
                    last_registration_diagnostics=latest_registration_diagnostics,
                    last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                )

            (
                next_crank_axis,
                current_diversity,
                max_diversity,
                current_scale,
                crank_hint,
            ) = _difficulty_guidance(
                graph=graph,
                task=latest_authoritative_task,
                bundle=latest_artifact_payload.artifacts if latest_artifact_payload is not None else None,
                available_atomic_tools=available_atomic_tools,
                history=difficulty_crank_history,
            )

            attempt_previous_outputs = dict(base_previous_outputs)
            request_kwargs = {
                "db_id": db_id,
                "atomic_tool_set_ref": atomic_tool_set_ref,
                "available_atomic_tools": available_atomic_tools,
                "domain_name": self.config.domain.name,
                "task_language": self.config.domain.language,
                "scenario_description": self.config.domain.scenario_description,
                "requested_category": requested_category,
                "attempt_index": attempt_index,
                "schema_summary": schema_summary,
                "memory": memory[-self.config.synthesis.runtime.explicit_memory_window :],
                "latest_registration_diagnostics": latest_registration_diagnostics,
                "latest_self_consistency_diagnostics": latest_self_consistency_diagnostics,
                "strongest_difficulty_vector": (
                    strongest_difficulty_vector or DifficultyVectorContract()
                ),
                "difficulty_crank_index": difficulty_crank_count,
                "difficulty_crank_history": list(difficulty_crank_history),
                "next_crank_axis": next_crank_axis,
                "current_diversity": current_diversity,
                "max_diversity": max_diversity,
                "current_scale": current_scale,
                "crank_hint": crank_hint,
                "latest_quality_gate_feedback": (
                    retry_seed.latest_quality_gate_feedback if retry_seed is not None else None
                ),
            }

            label_request = SynthesisStageRequest(
                phase=SynthesisPhase.LABEL_CONSTRUCTION,
                previous_outputs=attempt_previous_outputs,
                **request_kwargs,
            )
            label_result = await self._run_phase(label_request)
            assert isinstance(label_result.payload, LabelConstructionOutput)
            stage_results.append(label_result)
            memory.append(label_result.memory_entry)
            tool_traces.extend(label_result.tool_traces)
            attempt_previous_outputs[SynthesisPhase.LABEL_CONSTRUCTION] = label_result.payload

            task_request = SynthesisStageRequest(
                phase=SynthesisPhase.TASK_SYNTHESIS,
                previous_outputs=attempt_previous_outputs,
                **request_kwargs,
            )
            task_result = await self._run_phase(task_request)
            assert isinstance(task_result.payload, TaskSynthesisOutput)
            stage_results.append(task_result)
            memory.append(task_result.memory_entry)
            tool_traces.extend(task_result.tool_traces)
            attempt_previous_outputs[SynthesisPhase.TASK_SYNTHESIS] = task_result.payload

            authoritative_task = self._build_authoritative_task(
                requested_category=requested_category,
                label_output=label_result.payload,
                task_output=task_result.payload,
            )
            latest_authoritative_task = authoritative_task
            crank_request_active = retry_requires_harder
            current_difficulty_vector = authoritative_task.difficulty_vector
            weakened_axes = _weakened_difficulty_axes(
                previous=strongest_difficulty_vector,
                current=current_difficulty_vector,
            )
            strengthened_axes = _strengthened_difficulty_axes(
                previous=strongest_difficulty_vector,
                current=current_difficulty_vector,
            )
            original_strengthened_axes = list(strengthened_axes)
            projected_difficulty_vector = _project_retry_difficulty_vector(
                previous=strongest_difficulty_vector,
                current=current_difficulty_vector,
                requested_axis=label_request.next_crank_axis if crank_request_active else None,
            )
            if projected_difficulty_vector is not None:
                authoritative_task = authoritative_task.model_copy(
                    update={"difficulty_vector": projected_difficulty_vector}
                )
                latest_authoritative_task = authoritative_task
                label_result = label_result.model_copy(
                    update={
                        "payload": label_result.payload.model_copy(
                            update={"difficulty_vector": projected_difficulty_vector}
                        ),
                        "payload_repair_codes": [
                            *label_result.payload_repair_codes,
                            "difficulty_vector_projected_to_requested_axis",
                        ],
                    }
                )
                attempt_previous_outputs[SynthesisPhase.LABEL_CONSTRUCTION] = label_result.payload
                current_difficulty_vector = projected_difficulty_vector
                strengthened_axes = _strengthened_difficulty_axes(
                    previous=strongest_difficulty_vector,
                    current=current_difficulty_vector,
                )
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="difficulty_vector_projected",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "requested_axis": (
                            label_request.next_crank_axis.value
                            if label_request.next_crank_axis is not None
                            else None
                        ),
                    },
                    actual_data=self._actual_data_for_stage_result(label_result),
                    checks={
                        "original_strengthened_axes": [
                            axis.value for axis in original_strengthened_axes
                        ],
                        "projected_strengthened_axes": [axis.value for axis in strengthened_axes],
                    },
                    diagnostics={
                        "provider": label_result.provider,
                        "model": label_result.model,
                        "payload_repair_codes": list(label_result.payload_repair_codes),
                    },
                )
            if weakened_axes:
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="difficulty_weakened",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "monotonic_difficulty": True,
                        "strongest_difficulty_vector": difficulty_vector_json(
                            strongest_difficulty_vector or DifficultyVectorContract()
                        ),
                    },
                    actual_data=self._actual_data_for_stage_result(label_result),
                    checks={
                        "weakened_axes": list(weakened_axes),
                        "strengthened_axes": [axis.value for axis in strengthened_axes],
                    },
                    diagnostics={
                        "provider": label_result.provider,
                        "model": label_result.model,
                    },
                )
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.DIFFICULTY_WEAKENED,
                    provider=label_result.provider,
                    model=label_result.model,
                    memory_summary=label_result.memory_entry.summary,
                    error_message=(
                        "label construction reduced difficulty on axes: "
                        + ",".join(weakened_axes)
                    ),
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "label-first synthesis exhausted self-consistency budget after difficulty weakening",
                        attempts=attempts,
                        last_registration_diagnostics=latest_registration_diagnostics,
                        last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                    )
                continue
            if len(strengthened_axes) > 1 or (
                crank_request_active
                and len(strengthened_axes) == 1
                and label_request.next_crank_axis is not None
                and strengthened_axes[0] != label_request.next_crank_axis
            ) or (
                len(strengthened_axes) == 1
                and not _is_valid_difficulty_crank_step(
                    history=difficulty_crank_history,
                    strengthened_axes=strengthened_axes,
                )
            ):
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="difficulty_crank_invalid",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "max_strengthened_axes_per_attempt": 1,
                        "difficulty_crank_order": [axis.value for axis in DIFFICULTY_CRANK_ORDER],
                    },
                    actual_data=self._actual_data_for_stage_result(label_result),
                    checks={
                        "strengthened_axes": [axis.value for axis in strengthened_axes],
                        "strengthened_axis_count": len(strengthened_axes),
                        "difficulty_crank_history": [axis.value for axis in difficulty_crank_history],
                    },
                    diagnostics={
                        "provider": label_result.provider,
                        "model": label_result.model,
                    },
                )
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.DIFFICULTY_CRANK_INVALID,
                    provider=label_result.provider,
                    model=label_result.model,
                    memory_summary=label_result.memory_entry.summary,
                    error_message=(
                        "difficulty crank violated axis order: "
                        + ",".join(axis.value for axis in strengthened_axes)
                    ),
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "label-first synthesis exhausted self-consistency budget after invalid difficulty crank changes",
                        attempts=attempts,
                        last_registration_diagnostics=latest_registration_diagnostics,
                        last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                    )
                continue
            strongest_difficulty_vector = current_difficulty_vector

            artifact_request = SynthesisStageRequest(
                phase=SynthesisPhase.ARTIFACT_GENERATION,
                previous_outputs=attempt_previous_outputs,
                **request_kwargs,
            )
            artifact_result = await self._run_phase(artifact_request)
            assert isinstance(artifact_result.payload, ArtifactGenerationOutput)
            stage_results.append(artifact_result)
            memory.append(artifact_result.memory_entry)
            tool_traces.extend(artifact_result.tool_traces)

            artifact_payload, artifact_repair_codes = self._synchronize_artifact_payload_with_label(
                artifact_result.payload,
                authoritative_task=authoritative_task,
                instance_space=task_result.payload.instance_space,
            )
            latest_artifact_payload = artifact_payload
            if artifact_repair_codes:
                artifact_result = artifact_result.model_copy(
                    update={
                        "payload": artifact_payload,
                        "payload_repair_codes": [
                            *artifact_result.payload_repair_codes,
                            *artifact_repair_codes,
                        ],
                    }
                )

            try:
                registration_report, registration_diagnostics = await self._execute_registration_gate(
                    artifact_payload.artifacts,
                    proposed_environment=artifact_payload.proposed_environment,
                )
            except SynthesisRegistrationError as exc:
                self._emit_phase_monitor(
                    phase="registration_gate",
                    status="failed",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "required_artifacts": [
                            "solution",
                            "verifier",
                            "shadow_verifier",
                        ],
                    },
                    actual_data={
                        "status": (
                            exc.diagnostics.status.value
                            if exc.diagnostics is not None
                            else "failed"
                        ),
                        "failing_artifacts": (
                            [
                                artifact.value
                                for artifact in exc.diagnostics.failing_artifacts
                            ]
                            if exc.diagnostics is not None
                            else []
                        ),
                        "error_codes": (
                            list(exc.diagnostics.error_codes)
                            if exc.diagnostics is not None
                            else []
                        ),
                    },
                    checks={"passed": False},
                    diagnostics={
                        "provider": artifact_result.provider,
                        "model": artifact_result.model,
                    },
                )
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.REGISTRATION_FAILED,
                    provider=artifact_result.provider,
                    model=artifact_result.model,
                    memory_summary=artifact_result.memory_entry.summary,
                    error_message=str(exc),
                    registration_diagnostics=exc.diagnostics,
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                latest_registration_diagnostics = exc.diagnostics
                latest_self_consistency_diagnostics = None
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "label-first synthesis exhausted self-consistency budget after registration failures",
                        attempts=attempts,
                        last_registration_diagnostics=exc.diagnostics,
                    ) from exc
                continue

            self._emit_phase_monitor(
                phase="registration_gate",
                status="passed",
                expected_contract={
                    "attempt_index": attempt_index,
                    "required_artifacts": [
                        "solution",
                        "verifier",
                        "shadow_verifier",
                    ],
                },
                actual_data={
                    "status": registration_diagnostics.status.value,
                    "failing_artifacts": [
                        artifact.value for artifact in registration_diagnostics.failing_artifacts
                    ],
                    "error_codes": list(registration_diagnostics.error_codes),
                    "weak_signal_codes": list(registration_diagnostics.weak_signal_codes),
                },
                checks={
                    "passed": registration_diagnostics.status
                    is RegistrationBundleStatus.PASSED,
                },
                diagnostics={
                    "provider": artifact_result.provider,
                    "model": artifact_result.model,
                },
            )

            self_consistency_diagnostics = await self._execute_self_consistency_check(
                bundle=artifact_payload.artifacts,
                proposed_environment=artifact_payload.proposed_environment,
            )
            if artifact_result.payload_repair_codes:
                self_consistency_diagnostics = self_consistency_diagnostics.model_copy(
                    update={
                        "payload_repair_codes": list(
                            dict.fromkeys(
                                [
                                    *self_consistency_diagnostics.payload_repair_codes,
                                    *artifact_result.payload_repair_codes,
                                ]
                            )
                        )
                    }
                )
            self_consistency_diagnostics = self._reconcile_label_against_self_consistency(
                task=authoritative_task,
                label_output=label_result.payload,
                diagnostics=self_consistency_diagnostics,
            )
            if not self_consistency_diagnostics.passed:
                self._emit_phase_monitor(
                    phase="self_consistency",
                    status="failed",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "expected_fact_keys": list(
                            self_consistency_diagnostics.expected_fact_keys
                        ),
                    },
                    actual_data={
                        "answer_preview": self_consistency_diagnostics.answer,
                        "error_codes": list(self_consistency_diagnostics.error_codes),
                        "payload_repair_codes": list(
                            self_consistency_diagnostics.payload_repair_codes
                        ),
                        "solution_tool_calls": self_consistency_diagnostics.solution_tool_calls,
                        "verifier_tool_calls": self_consistency_diagnostics.verifier_tool_calls,
                        "shadow_verifier_tool_calls": (
                            self_consistency_diagnostics.shadow_verifier_tool_calls
                        ),
                        "verify_result": self_consistency_diagnostics.verify_result,
                        "shadow_verify_result": (
                            self_consistency_diagnostics.shadow_verify_result
                        ),
                    },
                    checks={
                        "passed": False,
                        "fact_keys_match": not (
                            self_consistency_diagnostics.missing_fact_keys
                            or self_consistency_diagnostics.extra_fact_keys
                        ),
                    },
                    diagnostics={
                        "provider": artifact_result.provider,
                        "model": artifact_result.model,
                    },
                )
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
                    provider=artifact_result.provider,
                    model=artifact_result.model,
                    memory_summary=artifact_result.memory_entry.summary,
                    error_message="solution output did not satisfy the primary verifier or planned label",
                    registration_diagnostics=registration_diagnostics,
                    self_consistency_diagnostics=self_consistency_diagnostics,
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                latest_registration_diagnostics = registration_diagnostics
                latest_self_consistency_diagnostics = self_consistency_diagnostics
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "label-first synthesis exhausted self-consistency budget after verifier failures",
                        attempts=attempts,
                        last_registration_diagnostics=registration_diagnostics,
                        last_self_consistency_diagnostics=self_consistency_diagnostics,
                    )
                continue

            self._emit_phase_monitor(
                phase="self_consistency",
                status="passed",
                expected_contract={
                    "attempt_index": attempt_index,
                    "expected_fact_keys": list(self_consistency_diagnostics.expected_fact_keys),
                },
                actual_data={
                    "answer_preview": self_consistency_diagnostics.answer,
                    "payload_repair_codes": list(
                        self_consistency_diagnostics.payload_repair_codes
                    ),
                    "solution_tool_calls": self_consistency_diagnostics.solution_tool_calls,
                    "verifier_tool_calls": self_consistency_diagnostics.verifier_tool_calls,
                    "shadow_verifier_tool_calls": (
                        self_consistency_diagnostics.shadow_verifier_tool_calls
                    ),
                    "verify_result": self_consistency_diagnostics.verify_result,
                    "shadow_verify_result": self_consistency_diagnostics.shadow_verify_result,
                },
                checks={
                    "passed": True,
                    "fact_keys_match": not (
                        self_consistency_diagnostics.missing_fact_keys
                        or self_consistency_diagnostics.extra_fact_keys
                    ),
                },
                diagnostics={
                    "provider": artifact_result.provider,
                    "model": artifact_result.model,
                },
            )

            try:
                (
                    instances,
                    canonical_answers,
                    materialized_cross_instance_set,
                ) = self._materialize_instances_and_canonical_answers(
                    proposed_environment=artifact_payload.proposed_environment,
                    canonical_answer_json=label_result.payload.canonical_answer_json,
                )
            except CanonicalizationError:
                self._emit_phase_monitor(
                    phase="canonical_materialization",
                    status="failed",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "minimum_instances": 1,
                    },
                    actual_data={},
                    checks={"canonicalization_passed": False},
                    diagnostics={
                        "provider": artifact_result.provider,
                        "model": artifact_result.model,
                        "error_code": "canonical_answer_schema_mismatch",
                    },
                )
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
                    provider=artifact_result.provider,
                    model=artifact_result.model,
                    memory_summary=artifact_result.memory_entry.summary,
                    error_message="planned canonical answer could not be canonicalized against the output schema",
                    registration_diagnostics=registration_diagnostics,
                    self_consistency_diagnostics=materialization_diagnostics,
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                latest_registration_diagnostics = registration_diagnostics
                latest_self_consistency_diagnostics = materialization_diagnostics
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "label-first synthesis exhausted self-consistency budget after canonical answer materialization failures",
                        attempts=attempts,
                        last_registration_diagnostics=registration_diagnostics,
                        last_self_consistency_diagnostics=materialization_diagnostics,
                    )
                continue

            self._emit_phase_monitor(
                phase="canonical_materialization",
                status="passed",
                expected_contract={
                    "attempt_index": attempt_index,
                    "minimum_instances": 1,
                },
                actual_data={
                    "instance_count": len(instances),
                    "canonical_answer_count": len(canonical_answers),
                    "rendered_user_prompts": [
                        instance.rendered_user_prompt for instance in instances
                    ],
                    "canonical_answer_jsons": [
                        answer.canonical_answer_json for answer in canonical_answers
                    ],
                    "solution_fingerprints": [
                        answer.solution_fingerprint for answer in canonical_answers
                    ],
                },
                checks={
                    "instance_count_matches_canonical_answers": len(instances)
                    == len(canonical_answers),
                    "minimum_instances_satisfied": len(instances)
                    >= materialized_cross_instance_set.minimum_required,
                },
                diagnostics={
                    "provider": artifact_result.provider,
                    "model": artifact_result.model,
                },
            )

            passed_attempt = SynthesisSelfConsistencyAttempt(
                attempt_index=attempt_index,
                outcome=SynthesisSelfConsistencyOutcome.PASSED,
                provider=artifact_result.provider,
                model=artifact_result.model,
                memory_summary=artifact_result.memory_entry.summary,
                registration_diagnostics=registration_diagnostics,
                self_consistency_diagnostics=self_consistency_diagnostics,
            )
            attempts.append(passed_attempt)
            retry_requires_harder = False
            return (
                artifact_payload.proposed_environment,
                artifact_payload.artifacts,
                registration_report,
                registration_diagnostics,
                self_consistency_diagnostics,
                instances,
                canonical_answers,
                materialized_cross_instance_set,
                attempts,
            )

        raise SynthesisSelfConsistencyError(
            "label-first synthesis exhausted self-consistency budget",
            attempts=attempts,
            last_registration_diagnostics=latest_registration_diagnostics,
            last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
        )

    async def _run_artifact_generation_with_self_consistency(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
        graph: SchemaGraph,
        schema_summary: dict[str, object],
        previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput],
        stage_results: list[SynthesisStageResult],
        memory: list[SynthesisMemoryEntry],
        tool_traces: list[SynthesisToolTraceEntry],
        atomic_tool_set_ref: str,
        available_atomic_tools: list[dict[str, object]],
        retry_seed: SynthesisDifficultyRetrySeed | None = None,
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
        latest_artifact_payload: ArtifactGenerationOutput | None = None
        strongest_difficulty_vector: DifficultyVectorContract | None = (
            retry_seed.strongest_difficulty_vector if retry_seed is not None else None
        )
        difficulty_crank_count = (
            retry_seed.difficulty_crank_index if retry_seed is not None else 0
        )
        difficulty_crank_history: list[DifficultyAxis] = list(
            retry_seed.difficulty_crank_history if retry_seed is not None else []
        )
        retry_requires_harder = (
            retry_seed.retry_requires_harder if retry_seed is not None else False
        )
        max_iterations = self.config.synthesis.runtime.max_self_consistency_iterations

        for attempt_index in range(1, max_iterations + 1):
            if (
                retry_requires_harder
                and difficulty_crank_count >= self.config.synthesis.runtime.max_difficulty_cranks
            ):
                next_limit_axis = _next_difficulty_crank_axis(history=difficulty_crank_history)
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="difficulty_crank_limit_exceeded",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "max_difficulty_cranks": (
                            self.config.synthesis.runtime.max_difficulty_cranks
                        ),
                    },
                    actual_data={
                        "difficulty_crank_index": difficulty_crank_count,
                        "difficulty_crank_history": [axis.value for axis in difficulty_crank_history],
                        "next_crank_axis": next_limit_axis.value,
                    },
                    checks={"retry_requires_harder": True},
                    diagnostics={},
                )
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.DIFFICULTY_CRANK_LIMIT_EXCEEDED,
                    provider="runtime",
                    model="runtime",
                    memory_summary="difficulty crank budget exhausted before next retry",
                    error_message=(
                        "difficulty crank limit exceeded: "
                        f"max={self.config.synthesis.runtime.max_difficulty_cranks}"
                    ),
                )
                attempts.append(attempt)
                raise SynthesisSelfConsistencyError(
                    "artifact generation exhausted self-consistency budget after exceeding the difficulty crank limit",
                    attempts=attempts,
                    last_registration_diagnostics=latest_registration_diagnostics,
                    last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                )
            (
                next_crank_axis,
                current_diversity,
                max_diversity,
                current_scale,
                crank_hint,
            ) = _difficulty_guidance(
                graph=graph,
                task=(
                    latest_artifact_payload.proposed_environment.task
                    if latest_artifact_payload is not None
                    else None
                ),
                bundle=latest_artifact_payload.artifacts if latest_artifact_payload is not None else None,
                available_atomic_tools=available_atomic_tools,
                history=difficulty_crank_history,
            )
            request = SynthesisStageRequest(
                phase=SynthesisPhase.ARTIFACT_GENERATION,
                db_id=db_id,
                atomic_tool_set_ref=atomic_tool_set_ref,
                available_atomic_tools=available_atomic_tools,
                domain_name=self.config.domain.name,
                scenario_description=self.config.domain.scenario_description,
                requested_category=requested_category,
                attempt_index=attempt_index,
                schema_summary=schema_summary,
                previous_outputs=previous_outputs,
                memory=memory[-self.config.synthesis.runtime.explicit_memory_window :],
                latest_registration_diagnostics=latest_registration_diagnostics,
                latest_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                strongest_difficulty_vector=(
                    strongest_difficulty_vector or DifficultyVectorContract()
                ),
                difficulty_crank_index=difficulty_crank_count,
                difficulty_crank_history=list(difficulty_crank_history),
                next_crank_axis=next_crank_axis,
                current_diversity=current_diversity,
                max_diversity=max_diversity,
                current_scale=current_scale,
                crank_hint=crank_hint,
                latest_quality_gate_feedback=(
                    retry_seed.latest_quality_gate_feedback if retry_seed is not None else None
                ),
            )
            result = await self._run_phase(request)
            assert isinstance(result.payload, ArtifactGenerationOutput)
            stage_results.append(result)
            memory.append(result.memory_entry)
            tool_traces.extend(result.tool_traces)

            artifact_payload = result.payload
            latest_artifact_payload = artifact_payload
            crank_request_active = retry_requires_harder
            current_difficulty_vector = artifact_payload.proposed_environment.task.difficulty_vector
            weakened_axes = _weakened_difficulty_axes(
                previous=strongest_difficulty_vector,
                current=current_difficulty_vector,
            )
            strengthened_axes = _strengthened_difficulty_axes(
                previous=strongest_difficulty_vector,
                current=current_difficulty_vector,
            )
            original_strengthened_axes = list(strengthened_axes)
            projected_difficulty_vector = _project_retry_difficulty_vector(
                previous=strongest_difficulty_vector,
                current=current_difficulty_vector,
                requested_axis=request.next_crank_axis if crank_request_active else None,
            )
            if projected_difficulty_vector is not None:
                projected_task = artifact_payload.proposed_environment.task.model_copy(
                    update={"difficulty_vector": projected_difficulty_vector}
                )
                projected_environment = artifact_payload.proposed_environment.model_copy(
                    update={"task": projected_task}
                )
                artifact_payload = artifact_payload.model_copy(
                    update={"proposed_environment": projected_environment}
                )
                latest_artifact_payload = artifact_payload
                result = result.model_copy(
                    update={
                        "payload": artifact_payload,
                        "payload_repair_codes": [
                            *result.payload_repair_codes,
                            "difficulty_vector_projected_to_requested_axis",
                        ],
                    }
                )
                current_difficulty_vector = projected_difficulty_vector
                strengthened_axes = _strengthened_difficulty_axes(
                    previous=strongest_difficulty_vector,
                    current=current_difficulty_vector,
                )
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="difficulty_vector_projected",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "requested_axis": (
                            request.next_crank_axis.value
                            if request.next_crank_axis is not None
                            else None
                        ),
                    },
                    actual_data=self._actual_data_for_stage_result(result),
                    checks={
                        "original_strengthened_axes": [
                            axis.value for axis in original_strengthened_axes
                        ],
                        "projected_strengthened_axes": [axis.value for axis in strengthened_axes],
                    },
                    diagnostics={
                        "provider": result.provider,
                        "model": result.model,
                        "payload_repair_codes": list(result.payload_repair_codes),
                    },
                )
            if weakened_axes:
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="difficulty_weakened",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "monotonic_difficulty": True,
                        "strongest_difficulty_vector": difficulty_vector_json(
                            strongest_difficulty_vector or DifficultyVectorContract()
                        ),
                    },
                    actual_data=self._actual_data_for_stage_result(result),
                    checks={
                        "weakened_axes": list(weakened_axes),
                        "strengthened_axes": [axis.value for axis in strengthened_axes],
                    },
                    diagnostics={
                        "provider": result.provider,
                        "model": result.model,
                    },
                )
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
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after difficulty weakening",
                        attempts=attempts,
                        last_registration_diagnostics=latest_registration_diagnostics,
                        last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                    )
                continue
            if len(strengthened_axes) > 1 or (
                crank_request_active
                and len(strengthened_axes) == 1
                and request.next_crank_axis is not None
                and strengthened_axes[0] != request.next_crank_axis
            ) or (
                len(strengthened_axes) == 1
                and not _is_valid_difficulty_crank_step(
                    history=difficulty_crank_history,
                    strengthened_axes=strengthened_axes,
                )
            ):
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="difficulty_crank_invalid",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "max_strengthened_axes_per_attempt": 1,
                        "difficulty_crank_order": [axis.value for axis in DIFFICULTY_CRANK_ORDER],
                    },
                    actual_data=self._actual_data_for_stage_result(result),
                    checks={
                        "strengthened_axes": [axis.value for axis in strengthened_axes],
                        "strengthened_axis_count": len(strengthened_axes),
                        "difficulty_crank_history": [axis.value for axis in difficulty_crank_history],
                    },
                    diagnostics={
                        "provider": result.provider,
                        "model": result.model,
                    },
                )
                attempt = SynthesisSelfConsistencyAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisSelfConsistencyOutcome.DIFFICULTY_CRANK_INVALID,
                    provider=result.provider,
                    model=result.model,
                    memory_summary=result.memory_entry.summary,
                    error_message=(
                        "difficulty crank violated axis order: "
                        + ",".join(axis.value for axis in strengthened_axes)
                    ),
                )
                attempts.append(attempt)
                memory.append(self._attempt_feedback_entry(attempt))
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after invalid difficulty crank changes",
                        attempts=attempts,
                        last_registration_diagnostics=latest_registration_diagnostics,
                        last_self_consistency_diagnostics=latest_self_consistency_diagnostics,
                    )
                continue
            if artifact_payload.proposed_environment.task.category != requested_category:
                self._emit_phase_monitor(
                    phase="artifact_policy",
                    status="category_mismatch",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "requested_category": requested_category.value,
                    },
                    actual_data=self._actual_data_for_stage_result(result),
                    checks={
                        "task_category_matches_requested": False,
                    },
                    diagnostics={
                        "provider": result.provider,
                        "model": result.model,
                    },
                )
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
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
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
                self._emit_phase_monitor(
                    phase="registration_gate",
                    status="failed",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "required_artifacts": [
                            "solution",
                            "verifier",
                            "shadow_verifier",
                        ],
                    },
                    actual_data={
                        "status": (
                            exc.diagnostics.status.value
                            if exc.diagnostics is not None
                            else "failed"
                        ),
                        "failing_artifacts": (
                            [
                                artifact.value
                                for artifact in exc.diagnostics.failing_artifacts
                            ]
                            if exc.diagnostics is not None
                            else []
                        ),
                        "error_codes": (
                            list(exc.diagnostics.error_codes)
                            if exc.diagnostics is not None
                            else []
                        ),
                    },
                    checks={
                        "passed": False,
                    },
                    diagnostics={
                        "provider": result.provider,
                        "model": result.model,
                    },
                )
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
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after registration failures",
                        attempts=attempts,
                        last_registration_diagnostics=exc.diagnostics,
                    ) from exc
                continue
            self._emit_phase_monitor(
                phase="registration_gate",
                status="passed",
                expected_contract={
                    "attempt_index": attempt_index,
                    "required_artifacts": [
                        "solution",
                        "verifier",
                        "shadow_verifier",
                    ],
                },
                actual_data={
                    "status": registration_diagnostics.status.value,
                    "failing_artifacts": [
                        artifact.value for artifact in registration_diagnostics.failing_artifacts
                    ],
                    "error_codes": list(registration_diagnostics.error_codes),
                    "weak_signal_codes": list(registration_diagnostics.weak_signal_codes),
                },
                checks={
                    "passed": registration_diagnostics.status
                    is RegistrationBundleStatus.PASSED,
                },
                diagnostics={
                    "provider": result.provider,
                    "model": result.model,
                },
            )

            self_consistency_diagnostics = await self._execute_self_consistency_check(
                bundle=artifact_payload.artifacts,
                proposed_environment=artifact_payload.proposed_environment,
            )
            if result.payload_repair_codes:
                self_consistency_diagnostics = self_consistency_diagnostics.model_copy(
                    update={
                        "payload_repair_codes": list(
                            dict.fromkeys(
                                [
                                    *self_consistency_diagnostics.payload_repair_codes,
                                    *result.payload_repair_codes,
                                ]
                            )
                        )
                    }
                )
            if not self_consistency_diagnostics.passed:
                self._emit_phase_monitor(
                    phase="self_consistency",
                    status="failed",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "expected_fact_keys": list(
                            self_consistency_diagnostics.expected_fact_keys
                        ),
                    },
                    actual_data={
                        "answer_preview": self_consistency_diagnostics.answer,
                        "error_codes": list(self_consistency_diagnostics.error_codes),
                        "payload_repair_codes": list(
                            self_consistency_diagnostics.payload_repair_codes
                        ),
                        "solution_tool_calls": self_consistency_diagnostics.solution_tool_calls,
                        "verifier_tool_calls": self_consistency_diagnostics.verifier_tool_calls,
                        "shadow_verifier_tool_calls": (
                            self_consistency_diagnostics.shadow_verifier_tool_calls
                        ),
                        "verify_result": self_consistency_diagnostics.verify_result,
                        "shadow_verify_result": (
                            self_consistency_diagnostics.shadow_verify_result
                        ),
                    },
                    checks={
                        "passed": False,
                        "fact_keys_match": not (
                            self_consistency_diagnostics.missing_fact_keys
                            or self_consistency_diagnostics.extra_fact_keys
                        ),
                    },
                    diagnostics={
                        "provider": result.provider,
                        "model": result.model,
                    },
                )
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
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after verifier failures",
                        attempts=attempts,
                        last_registration_diagnostics=registration_diagnostics,
                        last_self_consistency_diagnostics=self_consistency_diagnostics,
                    )
                continue
            self._emit_phase_monitor(
                phase="self_consistency",
                status="passed",
                expected_contract={
                    "attempt_index": attempt_index,
                    "expected_fact_keys": list(self_consistency_diagnostics.expected_fact_keys),
                },
                actual_data={
                    "answer_preview": self_consistency_diagnostics.answer,
                    "payload_repair_codes": list(
                        self_consistency_diagnostics.payload_repair_codes
                    ),
                    "solution_tool_calls": self_consistency_diagnostics.solution_tool_calls,
                    "verifier_tool_calls": self_consistency_diagnostics.verifier_tool_calls,
                    "shadow_verifier_tool_calls": (
                        self_consistency_diagnostics.shadow_verifier_tool_calls
                    ),
                    "verify_result": self_consistency_diagnostics.verify_result,
                    "shadow_verify_result": self_consistency_diagnostics.shadow_verify_result,
                },
                checks={
                    "passed": True,
                    "fact_keys_match": not (
                        self_consistency_diagnostics.missing_fact_keys
                        or self_consistency_diagnostics.extra_fact_keys
                    ),
                },
                diagnostics={
                    "provider": result.provider,
                    "model": result.model,
                },
            )

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
                self._emit_phase_monitor(
                    phase="canonical_materialization",
                    status="failed",
                    expected_contract={
                        "attempt_index": attempt_index,
                        "minimum_instances": 1,
                    },
                    actual_data={},
                    checks={
                        "canonicalization_passed": False,
                    },
                    diagnostics={
                        "provider": result.provider,
                        "model": result.model,
                        "error_code": "canonical_answer_schema_mismatch",
                    },
                )
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
                if crank_request_active:
                    difficulty_crank_count, difficulty_crank_history = _consume_difficulty_crank_attempt(
                        count=difficulty_crank_count,
                        history=difficulty_crank_history,
                    )
                retry_requires_harder = True
                if attempt_index >= max_iterations:
                    raise SynthesisSelfConsistencyError(
                        "artifact generation exhausted self-consistency budget after canonical answer materialization failures",
                        attempts=attempts,
                        last_registration_diagnostics=registration_diagnostics,
                        last_self_consistency_diagnostics=materialization_diagnostics,
                    )
                continue
            self._emit_phase_monitor(
                phase="canonical_materialization",
                status="passed",
                expected_contract={
                    "attempt_index": attempt_index,
                    "minimum_instances": 1,
                },
                actual_data={
                    "instance_count": len(instances),
                    "canonical_answer_count": len(canonical_answers),
                    "rendered_user_prompts": [
                        instance.rendered_user_prompt for instance in instances
                    ],
                    "canonical_answer_jsons": [
                        answer.canonical_answer_json for answer in canonical_answers
                    ],
                    "solution_fingerprints": [
                        answer.solution_fingerprint for answer in canonical_answers
                    ],
                },
                checks={
                    "instance_count_matches_canonical_answers": len(instances)
                    == len(canonical_answers),
                    "minimum_instances_satisfied": len(instances)
                    >= materialized_cross_instance_set.minimum_required,
                },
                diagnostics={
                    "provider": result.provider,
                    "model": result.model,
                },
            )

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
            retry_requires_harder = False
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
    def _build_authoritative_task(
        *,
        requested_category: CategoryTaxonomy,
        label_output: LabelConstructionOutput,
        task_output: TaskSynthesisOutput,
    ) -> TaskContract:
        return TaskContract(
            question=task_output.question,
            category=requested_category,
            output_schema=label_output.output_schema,
            constraint_summary=task_output.constraint_summary,
            difficulty_vector=label_output.difficulty_vector,
            instance_parameters=label_output.instance_parameters,
        )

    @staticmethod
    def _synchronize_artifact_payload_with_label(
        payload: ArtifactGenerationOutput,
        *,
        authoritative_task: TaskContract,
        instance_space: InstanceSpaceContract,
    ) -> tuple[ArtifactGenerationOutput, list[str]]:
        repair_codes: list[str] = []
        proposed_environment = payload.proposed_environment
        if proposed_environment.task != authoritative_task:
            proposed_environment = proposed_environment.model_copy(
                update={"task": authoritative_task}
            )
            repair_codes.append("artifact_task_overridden_from_task_synthesis")
        if proposed_environment.instance_space != instance_space:
            proposed_environment = proposed_environment.model_copy(
                update={"instance_space": instance_space}
            )
            repair_codes.append("artifact_instance_space_overridden_from_task_synthesis")
        if not repair_codes:
            return payload, repair_codes
        return payload.model_copy(update={"proposed_environment": proposed_environment}), repair_codes

    @staticmethod
    def _reconcile_label_against_self_consistency(
        *,
        task: TaskContract,
        label_output: LabelConstructionOutput,
        diagnostics: SynthesisSelfConsistencyDiagnostics,
    ) -> SynthesisSelfConsistencyDiagnostics:
        try:
            expected_answer = json.loads(label_output.canonical_answer_json)
            expected_canonical = canonicalize_output(task.output_schema, expected_answer)
            actual_input = SynthesisAgentRuntime._normalize_solution_answer(
                task.output_schema,
                diagnostics.answer,
            )
            actual_canonical = canonicalize_output(task.output_schema, actual_input)
        except (json.JSONDecodeError, CanonicalizationError):
            return diagnostics.model_copy(
                update={
                    "passed": False,
                    "error_codes": [
                        *diagnostics.error_codes,
                        "label_canonical_answer_mismatch",
                    ],
                }
            )
        if actual_canonical == expected_canonical:
            return diagnostics
        return diagnostics.model_copy(
            update={
                "passed": False,
                "error_codes": [
                    *diagnostics.error_codes,
                    "label_canonical_answer_mismatch",
                ],
            }
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
            if self_consistency_diagnostics.payload_repair_codes:
                detail_parts.append(
                    "payload_repairs="
                    + ",".join(self_consistency_diagnostics.payload_repair_codes)
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

    def _ensure_grounded_schema_exploration(
        self,
        request: SynthesisStageRequest,
        result: SynthesisStageResult,
    ) -> None:
        if not isinstance(result.payload, SchemaExplorationOutput):
            return
        grounded = bool(result.tool_traces) and bool(result.payload.sample_observations)
        if grounded:
            return
        self._emit_phase_monitor(
            phase=request.phase.value,
            status="ungrounded",
            expected_contract=self._expected_contract_for_request(request),
            actual_data=self._actual_data_for_stage_result(result),
            checks={
                "tool_traces_present": bool(result.tool_traces),
                "sample_observations_present": bool(result.payload.sample_observations),
            },
            diagnostics={
                "provider": result.provider,
                "model": result.model,
            },
        )
        raise SynthesisPhaseExecutionError(
            "schema exploration must inspect real DB rows through atomic tool calls before proceeding",
            phase=request.phase,
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

    async def _database_pools_for_tools(self) -> DatabasePools:
        if self._database_pools is None:
            self._database_pools = await DatabasePools.create(self.config.database)
        return self._database_pools

    async def _tool_executors_for_bundle(
        self,
        *,
        db_id: str,
        bundle: AtomicToolBundle,
    ) -> dict[str, ToolExecutor]:
        cached = self._tool_executor_cache.get(db_id)
        if cached is not None:
            return cached
        pools = await self._database_pools_for_tools()
        assert self._atomic_tool_materializer is not None
        materialization = self._atomic_tool_materializer.materialize_bundle(bundle)
        module = load_atomic_tool_module(
            materialization.source_path,
            module_name=f"rl_task_foundry_synthesis_atomic_tools_{db_id}",
        )
        resolved = {
            tool.name: bind_atomic_tool_executor(
                module=module,
                tool_name=tool.name,
                pools=pools,
            )
            for tool in bundle.tools
        }
        self._tool_executor_cache[db_id] = resolved
        return resolved

    async def _prime_phase_backends_with_atomic_tools(
        self,
        *,
        db_id: str,
        bundle: AtomicToolBundle,
    ) -> None:
        tool_definitions = bundle.actor_tool_definitions()
        tool_executors = await self._tool_executors_for_bundle(db_id=db_id, bundle=bundle)
        for backends in (self.phase_backends or {}).values():
            for backend in backends:
                binder = getattr(backend, "bind_atomic_tools", None)
                if callable(binder):
                    binder(
                        tool_definitions=tool_definitions,
                        tool_executors=tool_executors,
                    )

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
            database_execution_config=self.config.database.model_dump(mode="json"),
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
            database_execution_config=self.config.database.model_dump(mode="json"),
            pool=self._registration_pool,
            verifier_probe_specs=self._build_verifier_probe_specs(proposed_environment),
            answer_field_names=_top_level_answer_field_names(proposed_environment.task),
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
        canonical_answer_json: str,
    ) -> tuple[
        list[MaterializedInstanceRecord],
        list[MaterializedCanonicalAnswerRecord],
        CrossInstanceSet,
    ]:
        canonical_input = json.loads(canonical_answer_json)
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
