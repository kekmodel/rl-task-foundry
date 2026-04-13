"""Synthesis meta-agent runtime skeleton."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256

from pydantic import Field, model_validator

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.db import DatabasePools, ensure_database_pools
from rl_task_foundry.pipeline.provider_resilience import (
    ProviderCircuitBreaker,
    ProviderCircuitSnapshot,
)
from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.synthesis.contracts import (
    ConstraintKind,
    ConstraintSummaryItem,
    CrossInstanceSet,
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
    entity_slug_from_get_tool_name,
    is_person_like_identifier,
    normalize_topic,
)
from rl_task_foundry.synthesis.canonicalize import (
    CanonicalizationError,
    canonical_json,
    canonicalize_output,
)
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolBundle,
    AtomicToolFamily,
    AtomicToolGenerator,
)
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.phase_monitor import (
    PipelinePhaseMonitorLogger,
    default_phase_monitor_log_path,
)
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
from rl_task_foundry.synthesis.rendered_prompt_builder import build_rendered_user_prompt
from rl_task_foundry.synthesis.schema_inference import extract_output_schema_from_canonical
from rl_task_foundry.synthesis.submit_draft_tool import (
    SubmitDraftController,
    SubmitDraftErrorCode,
    SubmitDraftPayload,
)
from rl_task_foundry.synthesis.tool_runtime import (
    ToolExecutor,
    build_shuffle_seed,
    bind_atomic_tool_executor,
    load_atomic_tool_module,
    with_tool_shuffle_seed,
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
    provider_status: dict[str, SynthesisProviderStatus] = Field(default_factory=dict)


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
        phase: str | None = None,
        backend_failures: list[SynthesisBackendFailure] | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.backend_failures = tuple(backend_failures or [])


class SynthesisProviderUnavailableError(SynthesisRuntimeError):
    """Raised when all provider candidates are currently unavailable."""

    def __init__(self, message: str, *, phase: str | None = None) -> None:
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


def summarize_atomic_tool_surface(
    bundle: AtomicToolBundle,
    *,
    max_entity_surfaces: int = 24,
) -> dict[str, object]:
    entity_surfaces: list[dict[str, object]] = []
    self_anchor_surfaces: list[str] = []
    for tool in bundle.tools:
        if tool.family not in {
            AtomicToolFamily.T1_POINT_LOOKUP,
            AtomicToolFamily.T4_FK_TRAVERSAL,
        }:
            continue
        properties = tool.returns_schema.get("properties", {})
        if not isinstance(properties, dict):
            items = tool.returns_schema.get("items")
            if isinstance(items, dict):
                properties = items.get("properties", {})
        if not isinstance(properties, dict):
            continue
        field_names = [str(key) for key in properties.keys()]
        readable_fields = [name for name in field_names if not name.endswith("_id")]
        entity_surfaces.append(
            {
                "tool_name": tool.name,
                "family": tool.family.value,
                "field_names": field_names,
                "readable_fields": readable_fields,
                "id_only": len(readable_fields) == 0,
            }
        )
        entity_slug = entity_slug_from_get_tool_name(tool.name)
        if entity_slug is not None and is_person_like_identifier(entity_slug):
            self_anchor_surfaces.append(tool.name)
    return {
        "entity_surfaces": entity_surfaces[:max_entity_surfaces],
        "self_anchor_surfaces": self_anchor_surfaces[:max_entity_surfaces],
    }


def forbidden_question_tokens(schema_summary: dict[str, object]) -> frozenset[str]:
    tokens = {
        "select ",
        " from ",
        " join ",
        " order by",
        " group by",
        " limit ",
        " where ",
    }
    tables = schema_summary.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            qualified_name = str(table.get("qualified_name") or "")
            table_name = qualified_name.split(".")[-1]
            if "_" in table_name:
                tokens.add(table_name.lower())
            if "_" in qualified_name:
                tokens.add(qualified_name.lower())
    return frozenset(token for token in tokens if token)


def _snapshot_to_status(snapshot: ProviderCircuitSnapshot) -> SynthesisProviderStatus:
    return SynthesisProviderStatus(
        observed_at=datetime.now(timezone.utc),
        total_requests=snapshot.total_requests,
        failures=snapshot.failures,
        error_rate=snapshot.error_rate,
        available=snapshot.available,
        cooldown_remaining_s=snapshot.cooldown_remaining_s,
    )


@dataclass(slots=True)
class SynthesisAgentRuntime:
    """Single-db synthesis runtime with lock-protected shared state."""

    config: AppConfig
    synthesis_backends: list[object] | None = None
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
    _environment_orchestrator: object | None = field(default=None, init=False, repr=False)
    _category_state_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    _conversation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    phase_monitor: PipelinePhaseMonitorLogger | None = None
    _owns_phase_monitor: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.synthesis_backends is None:
            from rl_task_foundry.synthesis.backend_openai_agents import (
                OpenAIAgentsSynthesisBackend,
            )

            backend = OpenAIAgentsSynthesisBackend(
                model_ref=self.config.models.composer,
                provider_config=self.config.providers[self.config.models.composer.provider],
                runtime_config=self.config.synthesis.runtime,
                traces_dir=self.config.output.traces_dir / "synthesis",
            )
            self.synthesis_backends = [backend]
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
        if self._environment_orchestrator is None:
            from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentOrchestrator

            self._environment_orchestrator = EnvironmentOrchestrator(self.config)
        if self.phase_monitor is None:
            self.phase_monitor = PipelinePhaseMonitorLogger(
                phase_monitor_log_path=default_phase_monitor_log_path(
                    self.config.output.traces_dir
                ),
                flow_kind="synthesis_runtime",
                flow_id=build_flow_id("synthesis_runtime"),
            )
            self._owns_phase_monitor = True

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_topic: str,
        graph: SchemaGraph | None = None,
    ) -> SynthesisEnvironmentDraft:
        requested_topic = normalize_topic(requested_topic)
        await self._bind_db_id(db_id)
        await self._ensure_category_available(db_id, requested_topic)
        resolved_graph = graph if graph is not None else await self._introspect_graph()
        atomic_tool_bundle = await self._ensure_atomic_tool_bundle(
            db_id=db_id,
            graph=resolved_graph,
        )
        schema_summary = summarize_schema_graph(resolved_graph)
        tool_surface_summary = summarize_atomic_tool_surface(atomic_tool_bundle)
        shuffle_seed = build_shuffle_seed(
            "synthesis",
            db_id,
            requested_topic,
            datetime.now(timezone.utc).isoformat(),
        )
        controller = SubmitDraftController(
            config=self.config,
            requested_topic=requested_topic,
            environment_orchestrator=self._environment_orchestrator,
            build_draft=lambda payload: self._build_draft_from_submission(
                db_id=db_id,
                requested_topic=requested_topic,
                atomic_tool_bundle=atomic_tool_bundle,
                submission=payload,
                schema_summary=schema_summary,
            ),
            phase_monitor=self.phase_monitor,
            max_submissions=self.config.synthesis.runtime.max_generation_attempts,
            forbidden_question_tokens=forbidden_question_tokens(schema_summary),
            self_anchor_surface_names=tuple(
                str(name)
                for name in tool_surface_summary.get("self_anchor_surfaces", ())
                if isinstance(name, str) and name.strip()
            ),
        )
        # The shared backend instances hold mutable bindings for tools and the current
        # submit controller, so we keep one full synthesis conversation bound at a time.
        async with self._conversation_lock:
            await self._prime_synthesis_backends_with_context(
                db_id=db_id,
                bundle=atomic_tool_bundle,
                controller=controller,
                shuffle_seed=shuffle_seed,
            )
            conversation_result = await self._run_synthesis_conversation(
                db_id=db_id,
                requested_topic=requested_topic,
                schema_summary=schema_summary,
                tool_surface_summary=tool_surface_summary,
            )
        if controller.accepted_draft is None:
            attempts = self._generation_attempts_from_submit_records(
                controller=controller,
                provider=conversation_result.provider,
                model=conversation_result.model,
            )
            last_attempt = attempts[-1] if attempts else None
            diagnostics = SynthesisArtifactDiagnostics(
                error_codes=list(last_attempt.artifact_diagnostics.error_codes)
                if last_attempt is not None and last_attempt.artifact_diagnostics is not None
                else [],
            )
            await self._record_category_discard(
                db_id,
                requested_topic,
                outcome=last_attempt.outcome if last_attempt is not None else None,
                error_codes=list(diagnostics.error_codes),
            )
            raise SynthesisArtifactGenerationError(
                "single-agent synthesis did not produce an accepted draft before budget or turn exhaustion",
                attempts=attempts,
                last_artifact_diagnostics=diagnostics,
            )

        accepted_draft = controller.accepted_draft.model_copy(
            update={
                "schema_summary": schema_summary,
                "generation_attempts": self._generation_attempts_from_submit_records(
                    controller=controller,
                    provider=conversation_result.provider,
                    model=conversation_result.model,
                ),
                "provider_status": self.provider_status(),
            }
        )
        self._emit_phase_monitor(
            phase="synthesis_conversation",
            status="accepted",
            expected_contract={
                "requested_topic": requested_topic,
                "max_turns": self.config.synthesis.runtime.max_turns,
            },
            actual_data={
                "env_id": accepted_draft.environment.env_id,
                "selected_topic": accepted_draft.selected_topic,
                "final_output_text": conversation_result.final_output_text,
                "turn_count": conversation_result.turn_count,
                "tool_call_count": len(conversation_result.tool_calls),
            },
            checks={
                "accepted_draft_present": True,
                "submit_draft_called": "submit_draft" in conversation_result.tool_calls,
            },
            diagnostics={
                "provider": conversation_result.provider,
                "model": conversation_result.model,
                "shuffle_seed": shuffle_seed,
                "token_usage": conversation_result.token_usage,
                "transcript_ref": conversation_result.transcript_ref,
                "tool_trace_ref": conversation_result.tool_trace_ref,
            },
        )
        await self._reset_category_failure_state(db_id, requested_topic)
        return accepted_draft

    async def close(self) -> None:
        if self._database_pools is not None:
            await self._database_pools.close()
            self._database_pools = None
        self._atomic_tool_bundles.clear()
        self._tool_executor_cache.clear()
        if self._owns_phase_monitor and self.phase_monitor is not None:
            self.phase_monitor.close()
            self.phase_monitor = None

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

    async def _run_synthesis_conversation(
        self,
        *,
        db_id: str,
        requested_topic: str,
        schema_summary: dict[str, object],
        tool_surface_summary: dict[str, object],
    ) -> object:
        candidate_backends = self.synthesis_backends or []
        if not candidate_backends:
            raise SynthesisPhaseExecutionError(
                "no synthesis backends configured for single-agent synthesis"
            )

        errors: list[str] = []
        backend_failures: list[SynthesisBackendFailure] = []
        for backend in candidate_backends:
            breaker = self._breakers[backend.provider_name]
            if not breaker.is_available():
                continue
            try:
                result = await backend.run_synthesis(
                    db_id=db_id,
                    requested_topic=requested_topic,
                    domain_name=self.config.domain.name,
                    task_language=self.config.domain.language,
                    scenario_description=self.config.domain.scenario_description,
                    schema_summary=schema_summary,
                    tool_surface_summary=tool_surface_summary,
                    max_turns=self.config.synthesis.runtime.max_turns,
                )
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
                "single-agent synthesis failed across candidate providers: " + ", ".join(errors),
                phase=None,
                backend_failures=backend_failures,
            )
        raise SynthesisProviderUnavailableError(
            "all providers are in cooldown for single-agent synthesis",
            phase=None,
        )

    async def _prime_synthesis_backends_with_context(
        self,
        *,
        db_id: str,
        bundle: AtomicToolBundle,
        controller: SubmitDraftController,
        shuffle_seed: str,
    ) -> None:
        tool_definitions = bundle.actor_tool_definitions()
        base_tool_executors = await self._tool_executors_for_bundle(db_id=db_id, bundle=bundle)
        tool_executors = {
            name: with_tool_shuffle_seed(executor, shuffle_seed=shuffle_seed)
            for name, executor in base_tool_executors.items()
        }
        for backend in self.synthesis_backends or []:
            bind_atomic_tools = getattr(backend, "bind_atomic_tools", None)
            if callable(bind_atomic_tools):
                bind_atomic_tools(
                    tool_definitions=tool_definitions,
                    tool_executors=tool_executors,
                )
            bind_submit_draft_controller = getattr(backend, "bind_submit_draft_controller", None)
            if callable(bind_submit_draft_controller):
                bind_submit_draft_controller(controller)

    def _build_draft_from_submission(
        self,
        *,
        db_id: str,
        requested_topic: str,
        atomic_tool_bundle: AtomicToolBundle,
        submission: SubmitDraftPayload,
        schema_summary: dict[str, object],
    ) -> SynthesisEnvironmentDraft:
        selected_topic = normalize_topic(submission.topic)
        canonical_input = submission.canonical_answer
        output_schema = extract_output_schema_from_canonical(canonical_input)
        normalized_constraints = [
            ConstraintSummaryItem(
                key=item.key,
                kind=(
                    ConstraintKind(item.kind)
                    if item.kind in {kind.value for kind in ConstraintKind}
                    else ConstraintKind.OTHER
                ),
                summary=item.summary,
                hard=item.hard,
            )
            for item in submission.constraint_summary
        ]
        task = TaskContract(
            question=submission.question,
            topic=selected_topic,
            output_schema=output_schema,
            constraint_summary=normalized_constraints,
            difficulty_vector=submission.difficulty_vector,
            instance_parameters=dict(submission.anchor_entity),
        )
        (
            instances,
            canonical_answers,
            materialized_cross_instance_set,
        ) = self._materialize_instances_and_canonical_answers(
            task=task,
            instance_space=submission.instance_space,
            canonical_answer_input=canonical_input,
            anchor_entity=submission.anchor_entity,
        )
        materialized_at = datetime.now(timezone.utc)
        environment = self._materialize_environment(
            atomic_tool_bundle=atomic_tool_bundle,
            db_id=db_id,
            requested_topic=selected_topic,
            created_at=materialized_at,
            materialized_cross_instance_set=materialized_cross_instance_set,
            task=task,
            instance_space=submission.instance_space,
        )
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
            generation_attempts=[],
            provider_status={},
        )

    @staticmethod
    def _generation_attempts_from_submit_records(
        *,
        controller: SubmitDraftController,
        provider: str,
        model: str,
    ) -> list[SynthesisGenerationAttempt]:
        attempts: list[SynthesisGenerationAttempt] = []
        for record in controller.attempts:
            error_codes = list(record.error_codes)
            outcome = SynthesisGenerationOutcome.ARTIFACT_INVALID
            if record.outcome == "accepted":
                outcome = SynthesisGenerationOutcome.PASSED
            elif SubmitDraftErrorCode.REJECT_TOO_EASY.value in error_codes:
                outcome = SynthesisGenerationOutcome.DIFFICULTY_CRANK_INVALID
            elif SubmitDraftErrorCode.REJECT_TOO_HARD.value in error_codes:
                outcome = SynthesisGenerationOutcome.DIFFICULTY_WEAKENED
            attempts.append(
                SynthesisGenerationAttempt(
                    attempt_index=record.index,
                    outcome=outcome,
                    provider=provider,
                    model=model,
                    memory_summary=record.message,
                    error_message=None if record.outcome == "accepted" else record.message,
                    artifact_diagnostics=(
                        SynthesisArtifactDiagnostics(error_codes=error_codes)
                        if error_codes
                        else None
                    ),
                )
            )
        return attempts

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

    async def _database_pools_for_tools(self) -> DatabasePools:
        if self._database_pools is None:
            self._database_pools = await ensure_database_pools(
                self._database_pools,
                self.config.database,
            )
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
        canonical_answer_input: object,
        anchor_entity: dict[str, object],
    ) -> tuple[
        list[MaterializedInstanceRecord],
        list[MaterializedCanonicalAnswerRecord],
        CrossInstanceSet,
    ]:
        canonical_answer = canonicalize_output(
            task.output_schema,
            canonical_answer_input,
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
        normalized = canonical_json(payload, default=str)
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
