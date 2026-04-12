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
    TopicName,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    StrictModel,
    TaskContract,
    difficulty_vector_json,
    flatten_difficulty_vector,
    normalize_topic,
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
from rl_task_foundry.synthesis.rendered_prompt_builder import build_rendered_user_prompt
from rl_task_foundry.synthesis.schema_inference import extract_output_schema_from_canonical
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
        "topic",
        "atomic_tool_set_ref",
        "difficulty_vector",
        "created_at",
        "generator_version",
        "tool_signature",
        "task_signature",
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


class SynthesisProviderStatus(StrictModel):
    observed_at: datetime
    total_requests: int
    failures: int
    error_rate: float
    available: bool
    cooldown_remaining_s: float


class SynthesisCategoryStatus(StrictModel):
    db_id: str
    topic: str
    consecutive_discards: int
    backed_off: bool
    backoff_until: datetime | None = None
    backoff_remaining_s: float = 0.0
    last_outcome: SynthesisGenerationOutcome | None = None
    last_error_codes: list[str] = Field(default_factory=list)
    last_updated_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_category_key(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "topic" not in payload and "category" in payload:
            payload["topic"] = payload.pop("category")
        return payload

    @model_validator(mode="after")
    def _validate_timezones(self) -> SynthesisCategoryStatus:
        self.topic = normalize_topic(self.topic)
        if self.backoff_until is not None and self.backoff_until.tzinfo is None:
            raise ValueError("backoff_until must be timezone-aware")
        if self.last_updated_at is not None and self.last_updated_at.tzinfo is None:
            raise ValueError("last_updated_at must be timezone-aware")
        return self

    @property
    def category(self) -> TopicName:
        return TopicName(self.topic)


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
    candidate_topics: list[str] = Field(min_length=1)
    sample_observations: list[str] = Field(default_factory=list, min_length=1)
    memory_summary: str = "schema exploration completed"

    @model_validator(mode="after")
    def _validate_topics(self) -> SchemaExplorationOutput:
        self.candidate_topics = [normalize_topic(topic) for topic in self.candidate_topics]
        return self

    @property
    def candidate_categories(self) -> list[TopicName]:
        return [TopicName(topic) for topic in self.candidate_topics]


class CategoryInferenceOutput(StrictModel):
    selected_topic: str
    rationale: str
    memory_summary: str = "category inference completed"

    @model_validator(mode="after")
    def _validate_topic(self) -> CategoryInferenceOutput:
        self.selected_topic = normalize_topic(self.selected_topic)
        return self

    @property
    def selected_category(self) -> TopicName:
        return TopicName(self.selected_topic)


class LabelConstructionOutput(StrictModel):
    canonical_answer_json: str = Field(min_length=1)
    anchor_entity: dict[str, object] = Field(default_factory=dict)
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
    instance_space: InstanceSpaceContract
    cross_instance_set: CrossInstanceSet = Field(default_factory=CrossInstanceSet)


SynthesisPhaseOutput = (
    SchemaExplorationOutput
    | CategoryInferenceOutput
    | LabelConstructionOutput
    | TaskSynthesisOutput
)


class SynthesisStageRequest(StrictModel):
    phase: SynthesisPhase
    db_id: str
    atomic_tool_set_ref: str | None = None
    available_atomic_tools: list[dict[str, object]] = Field(default_factory=list)
    domain_name: str
    task_language: str = "ko"
    scenario_description: str
    requested_topic: str | None = None
    attempt_index: int = 1
    schema_summary: dict[str, object] = Field(default_factory=dict)
    previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput] = Field(default_factory=dict)
    memory: list[SynthesisMemoryEntry] = Field(default_factory=list)
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

    @property
    def requested_category(self) -> TopicName | None:
        if self.requested_topic is None:
            return None
        return TopicName(self.requested_topic)


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
    previous_label_signatures: list[str] = Field(default_factory=list)

    @property
    def previous_solution_fingerprints(self) -> list[str]:
        return self.previous_label_signatures


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
    label_signature: str

    @property
    def solution_fingerprint(self) -> str:
        return self.label_signature


class SynthesisEnvironmentDraft(StrictModel):
    created_at: datetime
    db_id: str
    requested_topic: str
    schema_summary: dict[str, object] = Field(default_factory=dict)
    selected_topic: str
    environment: EnvironmentContract
    atomic_tool_bundle: AtomicToolBundle
    instances: list[MaterializedInstanceRecord] = Field(default_factory=list)
    canonical_answers: list[MaterializedCanonicalAnswerRecord] = Field(default_factory=list)
    generation_attempts: list[SynthesisGenerationAttempt] = Field(default_factory=list)
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


class SynthesisGenerationOutcome(StrEnum):
    PASSED = "passed"
    CATEGORY_MISMATCH = "category_mismatch"
    ARTIFACT_INVALID = "artifact_invalid"
    DIFFICULTY_WEAKENED = "difficulty_weakened"
    DIFFICULTY_CRANK_INVALID = "difficulty_crank_invalid"
    DIFFICULTY_CRANK_LIMIT_EXCEEDED = "difficulty_crank_limit_exceeded"


class SynthesisArtifactDiagnostics(StrictModel):
    error_codes: list[str] = Field(default_factory=list)
    payload_repair_codes: list[str] = Field(default_factory=list)


class SynthesisGenerationAttempt(StrictModel):
    attempt_index: int
    outcome: SynthesisGenerationOutcome
    provider: str
    model: str
    memory_summary: str
    error_message: str | None = None
    artifact_diagnostics: SynthesisArtifactDiagnostics | None = None


class SynthesisArtifactGenerationError(SynthesisRuntimeError):
    """Raised when artifact generation exhausts its bounded retry budget."""

    def __init__(
        self,
        message: str,
        *,
        attempts: list[SynthesisGenerationAttempt],
        last_artifact_diagnostics: SynthesisArtifactDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_artifact_diagnostics = last_artifact_diagnostics


class SynthesisDbBindingError(SynthesisRuntimeError):
    """Raised when a runtime instance is reused for a different logical database id."""


class SynthesisCategoryBackoffError(SynthesisRuntimeError):
    """Raised when a db/category pair is temporarily backed off after repeated discards."""

    def __init__(
        self,
        message: str,
        *,
        db_id: str,
        topic: str,
        consecutive_discards: int,
        backoff_until: datetime,
        last_outcome: SynthesisGenerationOutcome | None = None,
        last_error_codes: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.db_id = db_id
        self.topic = topic
        self.consecutive_discards = consecutive_discards
        self.backoff_until = backoff_until
        self.last_outcome = last_outcome
        self.last_error_codes = list(last_error_codes or [])

    @property
    def category(self) -> TopicName:
        return TopicName(self.topic)


@dataclass(slots=True)
class _CategoryFailureState:
    consecutive_discards: int = 0
    backoff_until: datetime | None = None
    last_outcome: SynthesisGenerationOutcome | None = None
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
    available_atomic_tools: list[dict[str, object]],
    task: TaskContract | None,
) -> int:
    if task is None:
        return 0
    return min(
        max(1, int(task.difficulty_vector.search_cost)),
        max(1, len(graph.tables)),
    )


def _search_cost_max_diversity(graph: SchemaGraph) -> int:
    return max(1, len(graph.tables))


def _search_cost_scale(
    *,
    task: TaskContract | None,
    available_atomic_tools: list[dict[str, object]],
) -> int:
    if task is None:
        return 0
    return max(1, len(task.instance_parameters) or len(available_atomic_tools) // 16 or 1)


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
    available_atomic_tools: list[dict[str, object]],
    history: list[DifficultyAxis],
) -> tuple[DifficultyAxis, int, int, int, str]:
    axis = _next_difficulty_crank_axis(history=history)
    if axis is DifficultyAxis.SEARCH_COST:
        current_diversity = _search_cost_diversity(
            graph=graph,
            available_atomic_tools=available_atomic_tools,
            task=task,
        )
        max_diversity = _search_cost_max_diversity(graph)
        current_scale = _search_cost_scale(
            task=task,
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
    _atomic_tool_bundles: dict[str, AtomicToolBundle] = field(
        default_factory=dict, init=False, repr=False
    )
    _tool_executor_cache: dict[str, dict[str, ToolExecutor]] = field(
        default_factory=dict, init=False, repr=False
    )
    _database_pools: DatabasePools | None = field(default=None, init=False, repr=False)
    _bound_db_id: str | None = field(default=None, init=False, repr=False)
    _category_failures: dict[tuple[str, str], _CategoryFailureState] = field(
        default_factory=dict, init=False, repr=False
    )
    _bind_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _graph_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
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
        requested_topic: str,
        graph: SchemaGraph | None = None,
        retry_seed: SynthesisDifficultyRetrySeed | None = None,
    ) -> SynthesisEnvironmentDraft:
        requested_topic = normalize_topic(requested_topic)
        await self._bind_db_id(db_id)
        await self._ensure_category_available(db_id, requested_topic)
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
                    requested_topic=requested_topic,
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
            selected_topic = category_payload.selected_topic
            if selected_topic != requested_topic:
                raise SynthesisCategoryMismatchError(
                    "category inference result did not match the requested topic"
                )

            (
                task,
                instance_space,
                instances,
                canonical_answers,
                materialized_cross_instance_set,
                generation_attempts,
            ) = await self._run_label_first_generation_loop(
                db_id=db_id,
                requested_topic=requested_topic,
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
                atomic_tool_bundle=atomic_tool_bundle,
                db_id=db_id,
                requested_topic=requested_topic,
                created_at=materialized_at,
                materialized_cross_instance_set=materialized_cross_instance_set,
                task=task,
                instance_space=instance_space,
            )
            await self._reset_category_failure_state(db_id, requested_topic)

            return SynthesisEnvironmentDraft(
                created_at=materialized_at,
                db_id=db_id,
                requested_topic=requested_topic,
                schema_summary=schema_summary,
                selected_topic=selected_topic,
                environment=environment,
                atomic_tool_bundle=atomic_tool_bundle,
                instances=instances,
                canonical_answers=canonical_answers,
                generation_attempts=generation_attempts,
                stage_results=stage_results,
                memory=memory,
                tool_traces=tool_traces,
                provider_status=self.provider_status(),
            )
        except SynthesisCategoryMismatchError:
            await self._record_category_discard(
                db_id,
                requested_topic,
                outcome=SynthesisGenerationOutcome.CATEGORY_MISMATCH,
                error_codes=["category_mismatch"],
            )
            raise
        except SynthesisArtifactGenerationError as exc:
            last_attempt = exc.attempts[-1] if exc.attempts else None
            await self._record_category_discard(
                db_id,
                requested_topic,
                outcome=last_attempt.outcome if last_attempt is not None else None,
                error_codes=(
                    list(exc.last_artifact_diagnostics.error_codes)
                    if exc.last_artifact_diagnostics is not None
                    else []
                ),
            )
            raise

    async def close(self) -> None:
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

    def _expected_contract_for_request(
        self,
        request: SynthesisStageRequest,
    ) -> dict[str, object]:
        base = {
            "db_id": request.db_id,
            "phase": request.phase.value,
            "attempt_index": request.attempt_index,
            "requested_topic": request.requested_topic,
            "atomic_tool_count": len(request.available_atomic_tools),
            "memory_window_size": len(request.memory),
        }
        if request.phase == SynthesisPhase.SCHEMA_EXPLORATION:
            base.update(
                {
                    "required_fields": [
                        "domain_hypothesis",
                        "candidate_topics",
                        "sample_observations",
                        "memory_summary",
                    ],
                    "candidate_topics_min_length": 1,
                    "sample_observations_min_length": 1,
                    "must_use_live_atomic_tools": True,
                }
            )
            return base
        if request.phase == SynthesisPhase.CATEGORY_INFERENCE:
            base.update(
                {
                    "required_fields": [
                        "selected_topic",
                        "rationale",
                        "memory_summary",
                    ],
                    "must_match_requested_topic": request.requested_topic is not None,
                }
            )
            return base
        if request.phase == SynthesisPhase.LABEL_CONSTRUCTION:
            base.update(
                {
                    "required_fields": [
                        "canonical_answer_json",
                        "anchor_entity",
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
        return base

    def _actual_data_for_stage_result(
        self,
        result: SynthesisStageResult,
    ) -> dict[str, object]:
        payload = result.payload
        if isinstance(payload, SchemaExplorationOutput):
            return {
                "domain_hypothesis": payload.domain_hypothesis,
                "candidate_topics": list(payload.candidate_topics),
                "sample_observations": list(payload.sample_observations),
                "memory_summary": payload.memory_summary,
            }
        if isinstance(payload, CategoryInferenceOutput):
            return {
                "selected_topic": payload.selected_topic,
                "rationale": payload.rationale,
                "memory_summary": payload.memory_summary,
            }
        if isinstance(payload, LabelConstructionOutput):
            return {
                "canonical_answer_json": payload.canonical_answer_json,
                "anchor_entity": dict(payload.anchor_entity),
                "difficulty_vector": difficulty_vector_json(payload.difficulty_vector),
                "instance_parameter_keys": sorted(payload.instance_parameters.keys()),
                "label_summary": payload.label_summary,
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
        return {}

    def _checks_for_stage_result(
        self,
        request: SynthesisStageRequest,
        result: SynthesisStageResult,
    ) -> dict[str, object]:
        payload = result.payload
        if isinstance(payload, SchemaExplorationOutput):
            return {
                "domain_hypothesis_present": bool(payload.domain_hypothesis.strip()),
                "candidate_topics_non_empty": len(payload.candidate_topics) > 0,
                "sample_observations_non_empty": len(payload.sample_observations) > 0,
                "tool_traces_present": len(result.tool_traces) > 0,
                "requested_topic_in_candidates": (
                    request.requested_topic in payload.candidate_topics
                    if request.requested_topic is not None
                    else None
                ),
            }
        if isinstance(payload, CategoryInferenceOutput):
            return {
                "selected_topic_present": bool(payload.selected_topic),
                "selected_topic_matches_requested": (
                    payload.selected_topic == request.requested_topic
                    if request.requested_topic is not None
                    else None
                ),
                "rationale_present": bool(payload.rationale.strip()),
            }
        if isinstance(payload, LabelConstructionOutput):
            return {
                "canonical_answer_json_present": bool(payload.canonical_answer_json.strip()),
                "anchor_entity_present": bool(payload.anchor_entity),
                "label_summary_present": bool(payload.label_summary.strip()),
                "difficulty_vector_present": payload.difficulty_vector.total_score() >= 0.0,
            }
        if isinstance(payload, TaskSynthesisOutput):
            return {
                "question_present": bool(payload.question.strip()),
                "anchor_outputs_present": bool(payload.instance_space.anchor_query.outputs),
                "constraint_summary_present": len(payload.constraint_summary) > 0,
            }
        return {}

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
    ) -> dict[str, SynthesisCategoryStatus]:
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
            statuses: dict[str, SynthesisCategoryStatus] = {}
            for (state_db_id, topic), state in self._category_failures.items():
                if state_db_id != resolved_db_id:
                    continue
                remaining = 0.0
                if state.backoff_until is not None:
                    remaining = max(0.0, (state.backoff_until - now).total_seconds())
                statuses[topic] = SynthesisCategoryStatus(
                    db_id=state_db_id,
                    topic=topic,
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
        topic: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self._category_state_lock:
            state = self._category_failures.get((db_id, topic))
            if state is None:
                return
            if state.backoff_until is None:
                return
            if state.backoff_until <= now:
                del self._category_failures[(db_id, topic)]
                return
            raise SynthesisCategoryBackoffError(
                "synthesis category is temporarily backed off after repeated discards",
                db_id=db_id,
                topic=topic,
                consecutive_discards=state.consecutive_discards,
                backoff_until=state.backoff_until,
                last_outcome=state.last_outcome,
                last_error_codes=state.last_error_codes,
            )

    async def _record_category_discard(
        self,
        db_id: str,
        topic: str,
        *,
        outcome: SynthesisGenerationOutcome | None,
        error_codes: list[str] | None = None,
    ) -> None:
        async with self._category_state_lock:
            key = (db_id, topic)
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
        topic: str,
    ) -> None:
        async with self._category_state_lock:
            self._category_failures.pop((db_id, topic), None)

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

    async def _run_label_first_generation_loop(
        self,
        *,
        db_id: str,
        requested_topic: str,
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
        TaskContract,
        InstanceSpaceContract,
        list[MaterializedInstanceRecord],
        list[MaterializedCanonicalAnswerRecord],
        CrossInstanceSet,
        list[SynthesisGenerationAttempt],
    ]:
        attempts: list[SynthesisGenerationAttempt] = []
        latest_artifact_diagnostics: SynthesisArtifactDiagnostics | None = None
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
        max_iterations = self.config.synthesis.runtime.max_generation_attempts

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
                attempt = SynthesisGenerationAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisGenerationOutcome.DIFFICULTY_CRANK_LIMIT_EXCEEDED,
                    provider="runtime",
                    model="runtime",
                    memory_summary="difficulty crank budget exhausted before next retry",
                    error_message=(
                        "difficulty crank limit exceeded: "
                        f"max={self.config.synthesis.runtime.max_difficulty_cranks}"
                    ),
                )
                attempts.append(attempt)
                raise SynthesisArtifactGenerationError(
                    "label-first synthesis exhausted its bounded retry budget after exceeding the difficulty crank limit",
                    attempts=attempts,
                    last_artifact_diagnostics=latest_artifact_diagnostics,
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
                "requested_topic": requested_topic,
                "attempt_index": attempt_index,
                "schema_summary": schema_summary,
                "memory": memory[-self.config.synthesis.runtime.explicit_memory_window :],
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
                requested_topic=requested_topic,
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
                attempt = SynthesisGenerationAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisGenerationOutcome.DIFFICULTY_WEAKENED,
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
                    raise SynthesisArtifactGenerationError(
                        "label-first synthesis exhausted its bounded retry budget after difficulty weakening",
                        attempts=attempts,
                        last_artifact_diagnostics=latest_artifact_diagnostics,
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
                attempt = SynthesisGenerationAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisGenerationOutcome.DIFFICULTY_CRANK_INVALID,
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
                    raise SynthesisArtifactGenerationError(
                        "label-first synthesis exhausted its bounded retry budget after invalid difficulty crank changes",
                        attempts=attempts,
                        last_artifact_diagnostics=latest_artifact_diagnostics,
                    )
                continue
            strongest_difficulty_vector = current_difficulty_vector
            artifact_diagnostics = SynthesisArtifactDiagnostics(
                error_codes=[],
                payload_repair_codes=[
                    *label_result.payload_repair_codes,
                    *task_result.payload_repair_codes,
                ],
            )
            latest_artifact_diagnostics = artifact_diagnostics

            try:
                (
                    instances,
                    canonical_answers,
                    materialized_cross_instance_set,
                ) = self._materialize_instances_and_canonical_answers(
                    task=authoritative_task,
                    instance_space=task_result.payload.instance_space,
                    canonical_answer_json=label_result.payload.canonical_answer_json,
                    anchor_entity=label_result.payload.anchor_entity,
                )
            except (CanonicalizationError, json.JSONDecodeError):
                artifact_diagnostics = artifact_diagnostics.model_copy(
                    update={"error_codes": ["canonical_answer_schema_mismatch"]}
                )
                latest_artifact_diagnostics = artifact_diagnostics
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
                        "provider": task_result.provider,
                        "model": task_result.model,
                        "error_codes": list(artifact_diagnostics.error_codes),
                        "payload_repair_codes": list(artifact_diagnostics.payload_repair_codes),
                    },
                )
                attempt = SynthesisGenerationAttempt(
                    attempt_index=attempt_index,
                    outcome=SynthesisGenerationOutcome.ARTIFACT_INVALID,
                    provider=task_result.provider,
                    model=task_result.model,
                    memory_summary=task_result.memory_entry.summary,
                    error_message="planned canonical answer could not be canonicalized against the output schema",
                    artifact_diagnostics=artifact_diagnostics,
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
                    raise SynthesisArtifactGenerationError(
                        "label-first synthesis exhausted its bounded retry budget after canonical answer materialization failures",
                        attempts=attempts,
                        last_artifact_diagnostics=artifact_diagnostics,
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
                    "label_signatures": [
                        answer.label_signature for answer in canonical_answers
                    ],
                },
                checks={
                    "instance_count_matches_canonical_answers": len(instances)
                    == len(canonical_answers),
                    "minimum_instances_satisfied": len(instances)
                    >= materialized_cross_instance_set.minimum_required,
                },
                diagnostics={
                    "provider": task_result.provider,
                    "model": task_result.model,
                    "payload_repair_codes": list(artifact_diagnostics.payload_repair_codes),
                },
            )

            passed_attempt = SynthesisGenerationAttempt(
                attempt_index=attempt_index,
                outcome=SynthesisGenerationOutcome.PASSED,
                provider=task_result.provider,
                model=task_result.model,
                memory_summary=task_result.memory_entry.summary,
                artifact_diagnostics=artifact_diagnostics,
            )
            attempts.append(passed_attempt)
            retry_requires_harder = False
            return (
                authoritative_task,
                task_result.payload.instance_space,
                instances,
                canonical_answers,
                materialized_cross_instance_set,
                attempts,
            )

        raise SynthesisArtifactGenerationError(
            "label-first synthesis exhausted its bounded retry budget",
            attempts=attempts,
            last_artifact_diagnostics=latest_artifact_diagnostics,
        )

    @staticmethod
    def _build_authoritative_task(
        *,
        requested_topic: str,
        label_output: LabelConstructionOutput,
        task_output: TaskSynthesisOutput,
    ) -> TaskContract:
        canonical_answer = json.loads(label_output.canonical_answer_json)
        output_schema = extract_output_schema_from_canonical(canonical_answer)
        return TaskContract(
            question=task_output.question,
            topic=requested_topic,
            output_schema=output_schema,
            constraint_summary=task_output.constraint_summary,
            difficulty_vector=label_output.difficulty_vector,
            instance_parameters=label_output.instance_parameters,
        )

    @staticmethod
    def _attempt_feedback_entry(
        attempt: SynthesisGenerationAttempt,
    ) -> SynthesisMemoryEntry:
        detail_parts = [
            f"generation_attempt={attempt.attempt_index}",
            f"outcome={attempt.outcome.value}",
        ]
        if attempt.error_message:
            detail_parts.append(f"message={attempt.error_message}")
        if attempt.artifact_diagnostics is not None:
            if attempt.artifact_diagnostics.error_codes:
                detail_parts.append(
                    "errors=" + ",".join(attempt.artifact_diagnostics.error_codes)
                )
            if attempt.artifact_diagnostics.payload_repair_codes:
                detail_parts.append(
                    "payload_repairs="
                    + ",".join(attempt.artifact_diagnostics.payload_repair_codes)
                )
        return SynthesisMemoryEntry(
            phase=SynthesisPhase.TASK_SYNTHESIS,
            provider="runtime",
            model="generation_loop",
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

    def _materialize_environment(
        self,
        *,
        atomic_tool_bundle: AtomicToolBundle,
        db_id: str,
        requested_topic: str,
        created_at: datetime,
        materialized_cross_instance_set: CrossInstanceSet,
        task: TaskContract,
        instance_space: InstanceSpaceContract,
    ) -> EnvironmentContract:
        task_payload = task.model_dump(mode="python")
        task_signature = self._signature_for_payload(task_payload)
        tool_signature = self._signature_for_text(atomic_tool_bundle.source)
        env_id = self._build_env_id(
            db_id=db_id,
            topic=requested_topic,
            task_signature=task_signature,
            tool_signature=tool_signature,
        )
        payload = {
            "instance_space": instance_space.model_dump(mode="python"),
            "cross_instance_set": materialized_cross_instance_set.model_dump(mode="python"),
            "task": task.model_dump(mode="python"),
        }
        payload.update(
            {
                "env_id": env_id,
                "db_id": db_id,
                "domain": self.config.domain.name,
                "topic": requested_topic,
                "atomic_tool_set_ref": f"db://{db_id}",
                "difficulty_vector": task.difficulty_vector,
                "created_at": created_at,
                "generator_version": CURRENT_SYNTHESIS_GENERATOR_VERSION,
                "tool_signature": tool_signature,
                "task_signature": task_signature,
                "status": EnvironmentStatus.DRAFT,
                "quality_metrics": EnvironmentQualityMetrics().model_dump(mode="python"),
                "rollout_constraints": self._build_rollout_constraints().model_dump(
                    mode="python"
                ),
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
        task: TaskContract,
        instance_space: InstanceSpaceContract,
        canonical_answer_json: str,
        anchor_entity: dict[str, object],
    ) -> tuple[
        list[MaterializedInstanceRecord],
        list[MaterializedCanonicalAnswerRecord],
        CrossInstanceSet,
    ]:
        canonical_input = json.loads(canonical_answer_json)
        canonical_answer = canonicalize_output(
            task.output_schema,
            canonical_input,
        )
        canonical_answer_json = canonical_json(canonical_answer)
        label_signature = self._signature_for_text(canonical_answer_json)
        instance_id = "instance_0001"
        params = dict(task.instance_parameters)
        instance = MaterializedInstanceRecord(
            instance_id=instance_id,
            rendered_user_prompt=build_rendered_user_prompt(
                task,
                anchor_entity=anchor_entity,
                canonical_answer=canonical_answer,
            ),
            params=params,
            anchor_values=dict(anchor_entity),
        )
        canonical_record = MaterializedCanonicalAnswerRecord(
            instance_id=instance_id,
            canonical_answer=canonical_answer,
            canonical_answer_json=canonical_answer_json,
            label_signature=label_signature,
        )
        cross_instance_set = CrossInstanceSet(
            minimum_required=1,
            require_distinct_label_signatures=True,
            instances=[
                InstanceContract(
                    instance_id=instance_id,
                    anchor_values=dict(anchor_entity),
                    parameter_values=params,
                    expected_label_signature=label_signature,
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
        topic: str,
        task_signature: str,
        tool_signature: str,
    ) -> str:
        digest = sha256(
            f"{db_id}|{topic}|{task_signature}|{tool_signature}".encode("utf-8")
        ).hexdigest()[:16]
        return f"env_{topic}_{digest}"
