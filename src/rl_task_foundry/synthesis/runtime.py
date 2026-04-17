"""Synthesis meta-agent runtime skeleton."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import Field, model_validator

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.db import DatabasePools
from rl_task_foundry.pipeline.provider_resilience import (
    ProviderCircuitBreaker,
    ProviderCircuitSnapshot,
)
from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.profiler import DataProfile
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolBundle,
    AtomicToolFamily,
)
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import (
    RolloutConstraintsContract,
    RuntimeValue,
    StrictModel,
    TaskBundleContract,
    TaskBundleStatus,
    TaskContract,
    TaskQualityMetrics,
    TopicName,
    normalize_topic,
)
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
from rl_task_foundry.synthesis.synthesis_db import SynthesisDb
from rl_task_foundry.synthesis.tool_runtime import (
    ToolExecutor,
    build_shuffle_seed,
    with_tool_shuffle_seed,
)

if TYPE_CHECKING:
    from rl_task_foundry.pipeline.solver_orchestrator import SolverOrchestrator
    from rl_task_foundry.synthesis.backend_openai_agents import SynthesisConversationResult


class _SynthesisBackendProtocol(Protocol):
    @property
    def provider_name(self) -> str: ...
    @property
    def model_name(self) -> str: ...
    def bind_atomic_tools(
        self,
        *,
        tool_definitions: list[dict[str, Any]],
        tool_executors: dict[str, ToolExecutor],
    ) -> None: ...
    def bind_submit_draft_controller(self, controller: SubmitDraftController) -> None: ...
    async def run_synthesis(
        self,
        *,
        db_id: str,
        requested_topic: str | None,
        domain_name: str,
        task_language: str,
        scenario_description: str,
        schema_summary: dict[str, object],
        tool_surface_summary: dict[str, object],
        anchor_hint: dict[str, object] | None = None,
        data_profile: DataProfile | None = None,
        max_turns: int,
    ) -> SynthesisConversationResult: ...


CURRENT_SYNTHESIS_GENERATOR_VERSION = "milestone-3-runtime-v1"
RUNTIME_OWNED_TASK_BUNDLE_FIELDS = frozenset(
    {
        "task_id",
        "db_id",
        "domain",
        "topic",
        "atomic_tool_set_ref",
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


class SynthesisTaskDraft(StrictModel):
    created_at: datetime
    db_id: str
    requested_topic: str | None = None
    schema_summary: dict[str, object] = Field(default_factory=dict)
    selected_topic: str
    task_bundle: TaskBundleContract
    atomic_tool_bundle: AtomicToolBundle
    rendered_user_prompt: str
    anchor_entity: dict[str, object] = Field(default_factory=dict)
    canonical_answer_json: str
    label_signature: str
    generation_attempts: list[SynthesisGenerationAttempt] = Field(default_factory=list)
    provider_status: dict[str, SynthesisProviderStatus] = Field(default_factory=dict)

    @property
    def canonical_answer(self) -> object:
        return json.loads(self.canonical_answer_json)


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
SynthesisTaskDraft.model_rebuild()


def summarize_schema_graph(
    graph: SchemaGraph,
    *,
    max_tables: int,
) -> dict[str, object]:
    table_summaries: list[dict[str, object]] = []
    limited_tables = graph.tables[:max_tables]
    for table in limited_tables:
        columns = table.columns
        readable = [
            c.column_name
            for c in columns
            if not c.is_primary_key
            and not c.is_foreign_key
            and not c.column_name.endswith("_id")
        ]
        fk_cols = [
            c.column_name for c in columns if c.is_foreign_key
        ]
        outbound = graph.edges_from(
            table.table_name, schema_name=table.schema_name
        )
        inbound = graph.edges_to(
            table.table_name, schema_name=table.schema_name
        )
        surface = "readable" if readable else "id-only"
        table_summaries.append(
            {
                "qualified_name": table.qualified_name,
                "row_estimate": table.row_estimate,
                "primary_key": list(table.primary_key),
                "column_names": [
                    c.column_name for c in columns
                ],
                "readable_columns": readable,
                "fk_columns": fk_cols,
                "surface": surface,
                "outbound_edges": [
                    f"{e.source_qualified_name}"
                    f"->{e.target_qualified_name}"
                    for e in outbound
                ],
                "inbound_edges": [
                    f"{e.source_qualified_name}"
                    f"->{e.target_qualified_name}"
                    for e in inbound
                ],
                "fanout_in": [
                    {
                        "from": e.source_qualified_name,
                        "fanout": round(e.fanout_estimate, 1)
                        if e.fanout_estimate
                        else None,
                    }
                    for e in inbound
                    if e.fanout_estimate
                    and e.fanout_estimate > 1.5
                ],
            }
        )
    # identify hub tables (high inbound degree)
    hub_tables = [
        t["qualified_name"]
        for t in table_summaries
        if len(t.get("inbound_edges") or []) >= 3  # type: ignore[arg-type]
    ]
    # identify bridge tables (id-only, 2+ FK columns)
    bridge_tables = [
        t["qualified_name"]
        for t in table_summaries
        if t.get("surface") == "id-only"
        and len(t.get("fk_columns") or []) >= 2  # type: ignore[arg-type]
    ]
    return {
        "table_count": len(graph.tables),
        "edge_count": len(graph.edges),
        "included_table_count": len(limited_tables),
        "truncated": len(limited_tables) != len(graph.tables),
        "tables": table_summaries,
        "hub_tables": hub_tables,
        "bridge_tables": bridge_tables,
    }


def summarize_atomic_tool_surface(
    bundle: AtomicToolBundle,
    *,
    max_entity_surfaces: int,
) -> dict[str, object]:
    entity_surfaces: list[dict[str, object]] = []
    family_counts: dict[str, int] = {}
    for tool in bundle.tools:
        family_counts[tool.family.value] = family_counts.get(tool.family.value, 0) + 1
        if tool.family is not AtomicToolFamily.GET:
            continue
        properties = _tool_return_properties(tool.returns_schema)
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
    entity_surfaces.sort(
        key=lambda item: (
            bool(item.get("id_only")),
            str(item.get("tool_name", "")),
        )
    )
    return {
        "tool_count": len(bundle.tools),
        "family_counts": family_counts,
        "entity_surfaces": entity_surfaces[:max_entity_surfaces],
    }


def _tool_return_properties(returns_schema: dict[str, object]) -> dict[str, object] | None:
    properties = returns_schema.get("properties")
    if isinstance(properties, dict):
        return properties
    items = returns_schema.get("items")
    if isinstance(items, dict):
        item_properties = items.get("properties")
        if isinstance(item_properties, dict):
            return item_properties
    any_of = returns_schema.get("anyOf")
    if isinstance(any_of, list):
        for variant in any_of:
            if not isinstance(variant, dict):
                continue
            variant_properties = variant.get("properties")
            if isinstance(variant_properties, dict):
                return variant_properties
    return None


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
    """Single-db synthesis runtime.

    All per-database state (schema introspection, data profile, atomic tool
    bundle, tool executors, anchor sampler, DB pools) lives in the injected
    or lazily-constructed ``SynthesisDb``. The runtime itself only owns
    per-conversation orchestration state (provider breakers, category-failure
    backoff state, conversation lock, optional phase monitor).
    """

    config: AppConfig
    synthesis_backends: list[_SynthesisBackendProtocol] | None = None
    database_pools: DatabasePools | None = None
    solver_orchestrator: SolverOrchestrator | None = None
    synthesis_db: SynthesisDb | None = None
    phase_monitor: PipelinePhaseMonitorLogger | None = None
    _breakers: dict[str, ProviderCircuitBreaker] = field(
        default_factory=dict, init=False, repr=False
    )
    _synthesis_db: SynthesisDb | None = field(default=None, init=False, repr=False)
    _owns_synthesis_db: bool = field(default=True, init=False, repr=False)
    _solver_orchestrator: SolverOrchestrator | None = field(default=None, init=False, repr=False)
    _owns_solver_orchestrator: bool = field(default=True, init=False, repr=False)
    _bound_db_id: str | None = field(default=None, init=False, repr=False)
    _category_failures: dict[tuple[str, str], _CategoryFailureState] = field(
        default_factory=dict, init=False, repr=False
    )
    _bind_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _category_state_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _conversation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
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
                minimum_request_count=self.config.provider_resilience.minimum_request_count,
            )
            for provider_name in self.config.providers
        }
        if self.synthesis_db is not None:
            self._synthesis_db = self.synthesis_db
            self._owns_synthesis_db = False
            self._bound_db_id = self.synthesis_db.db_id
        if self.solver_orchestrator is not None:
            self._solver_orchestrator = self.solver_orchestrator
            self._owns_solver_orchestrator = False
        else:
            from rl_task_foundry.pipeline.solver_orchestrator import SolverOrchestrator

            self._solver_orchestrator = SolverOrchestrator(
                self.config,
                database_pools=self.database_pools,
            )
            self._owns_solver_orchestrator = True
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
        requested_topic: str | None = None,
        graph: SchemaGraph | None = None,
    ) -> SynthesisTaskDraft:
        if requested_topic:
            requested_topic = normalize_topic(requested_topic)
        await self._bind_db_id(db_id)
        synthesis_db = self._ensure_synthesis_db(db_id)
        if graph is not None:
            synthesis_db.adopt_schema_graph(graph)
        if requested_topic:
            await self._ensure_category_available(
                db_id, requested_topic
            )
        resolved_graph = await synthesis_db.schema_graph()
        data_profile = await synthesis_db.data_profile()
        atomic_tool_bundle = await synthesis_db.atomic_tool_bundle()
        schema_summary = summarize_schema_graph(
            resolved_graph,
            max_tables=self.config.synthesis.runtime.schema_summary_max_tables,
        )
        tool_surface_summary = summarize_atomic_tool_surface(
            atomic_tool_bundle,
            max_entity_surfaces=self.config.synthesis.runtime.tool_surface_summary_max_entries,
        )
        anchor_hint = await synthesis_db.random_anchor()
        shuffle_seed = build_shuffle_seed(
            "synthesis",
            db_id,
            requested_topic or "",
            datetime.now(timezone.utc).isoformat(),
        )
        assert self._solver_orchestrator is not None, "SolverOrchestrator not initialized"
        controller = SubmitDraftController(
            config=self.config,
            requested_topic=requested_topic,
            solver_orchestrator=self._solver_orchestrator,
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
                anchor_hint=anchor_hint,
                data_profile=data_profile,
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
            if requested_topic is not None:
                await self._record_category_discard(
                    db_id,
                    requested_topic,
                    outcome=last_attempt.outcome if last_attempt is not None else None,
                    error_codes=list(diagnostics.error_codes),
                )
            raise SynthesisArtifactGenerationError(
                "single-agent synthesis did not produce an accepted"
                " draft before budget or turn exhaustion",
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
                "task_id": accepted_draft.task_bundle.task_id,
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
        if requested_topic:
            await self._reset_category_failure_state(
                db_id, requested_topic
            )
        return accepted_draft

    async def close(self) -> None:
        if self._owns_solver_orchestrator and self._solver_orchestrator is not None:
            await self._solver_orchestrator.close()
        self._solver_orchestrator = None
        if self._owns_synthesis_db and self._synthesis_db is not None:
            await self._synthesis_db.close()
        self._synthesis_db = None
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
        requested_topic: str | None,
        schema_summary: dict[str, object],
        tool_surface_summary: dict[str, object],
        anchor_hint: dict[str, object] | None = None,
        data_profile: DataProfile | None = None,
    ) -> SynthesisConversationResult:
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
                    anchor_hint=anchor_hint,
                    data_profile=data_profile,
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
        synthesis_db = self._ensure_synthesis_db(db_id)
        base_tool_executors = await synthesis_db.tool_executors()
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
        requested_topic: str | None,
        atomic_tool_bundle: AtomicToolBundle,
        submission: SubmitDraftPayload,
        schema_summary: dict[str, object],
    ) -> SynthesisTaskDraft:
        selected_topic = normalize_topic(submission.topic)
        canonical_input = submission.canonical_answer
        output_schema = extract_output_schema_from_canonical(canonical_input)
        task = TaskContract(
            question=submission.question,
            topic=selected_topic,
            output_schema=output_schema,
            constraint_summary=[],
            instance_parameters=cast(dict[str, RuntimeValue], submission.parsed_entity),
        )
        canonical_answer_json = canonical_json(canonical_input, default=str)
        label_signature = self._signature_for_text(canonical_answer_json)
        rendered_user_prompt = build_rendered_user_prompt(
            task,
            anchor_entity=submission.parsed_entity,
            canonical_answer=canonical_input,
        )
        materialized_at = datetime.now(timezone.utc)
        task_bundle = self._materialize_task_bundle(
            atomic_tool_bundle=atomic_tool_bundle,
            db_id=db_id,
            selected_topic=selected_topic,
            created_at=materialized_at,
            task=task,
        )
        return SynthesisTaskDraft(
            created_at=materialized_at,
            db_id=db_id,
            requested_topic=requested_topic,
            schema_summary=schema_summary,
            selected_topic=selected_topic,
            task_bundle=task_bundle,
            atomic_tool_bundle=atomic_tool_bundle,
            rendered_user_prompt=rendered_user_prompt,
            anchor_entity=dict(submission.parsed_entity),
            canonical_answer_json=canonical_answer_json,
            label_signature=label_signature,
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

    def _ensure_synthesis_db(self, db_id: str) -> SynthesisDb:
        if self._synthesis_db is not None:
            if self._synthesis_db.db_id != db_id:
                raise SynthesisDbBindingError(
                    "SynthesisAgentRuntime is bound to db_id="
                    f"{self._synthesis_db.db_id}, refusing request for {db_id}"
                )
            return self._synthesis_db
        self._synthesis_db = SynthesisDb(
            db_id=db_id,
            config=self.config,
            database_pools=self.database_pools,
        )
        self._owns_synthesis_db = True
        return self._synthesis_db

    def _materialize_task_bundle(
        self,
        *,
        atomic_tool_bundle: AtomicToolBundle,
        db_id: str,
        selected_topic: str,
        created_at: datetime,
        task: TaskContract,
    ) -> TaskBundleContract:
        task_payload = task.model_dump(mode="python")
        task_signature = self._signature_for_payload(task_payload)
        tool_signature = self._signature_for_text(atomic_tool_bundle.source)
        task_id = self._build_task_id(
            db_id=db_id,
            topic=selected_topic,
            task_signature=task_signature,
            tool_signature=tool_signature,
        )
        payload: dict[str, Any] = {"task": task.model_dump(mode="python")}
        payload.update(
            {
                "task_id": task_id,
                "db_id": db_id,
                "domain": self.config.domain.name,
                "topic": selected_topic,
                "atomic_tool_set_ref": f"db://{db_id}",
                "created_at": created_at,
                "generator_version": CURRENT_SYNTHESIS_GENERATOR_VERSION,
                "tool_signature": tool_signature,
                "task_signature": task_signature,
                "status": TaskBundleStatus.DRAFT,
                "quality_metrics": TaskQualityMetrics().model_dump(mode="python"),
                "rollout_constraints": self._build_rollout_constraints().model_dump(mode="python"),
            }
        )
        return TaskBundleContract.model_validate(payload)

    def _build_rollout_constraints(self) -> RolloutConstraintsContract:
        return RolloutConstraintsContract(
            max_turns=self.config.solver_runtime.max_turns,
            max_episode_duration_ms=(
                self.config.database.statement_timeout_ms * self.config.solver_runtime.max_turns
            ),
            max_tool_rows=self.config.atomic_tools.bounded_result_limit,
        )

    async def _bind_db_id(self, db_id: str) -> None:
        async with self._bind_lock:
            if self._bound_db_id is None:
                self._bound_db_id = db_id
                return
            if self._bound_db_id != db_id:
                raise SynthesisDbBindingError(
                    "SynthesisAgentRuntime instances are single-db; "
                    "create a new runtime per db_id"
                )

    @staticmethod
    def _signature_for_text(source: str) -> str:
        return f"sha256:{sha256(source.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _signature_for_payload(payload: dict[str, object]) -> str:
        normalized = canonical_json(payload, default=str)
        return f"sha256:{sha256(normalized.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _build_task_id(
        *,
        db_id: str,
        topic: str,
        task_signature: str,
        tool_signature: str,
    ) -> str:
        digest = sha256(
            f"{db_id}|{topic}|{task_signature}|{tool_signature}".encode("utf-8")
        ).hexdigest()[:16]
        return f"task_{topic}_{digest}"
