"""Synthesis meta-agent runtime skeleton."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Protocol

from pydantic import Field

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
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnvironmentStatus,
    InstanceSpaceContract,
    OutputFieldContract,
    OutputFieldType,
    ShadowVerifierContract,
    SolutionContract,
    StrictModel,
    TaskContract,
    ToolContract,
    ToolSelfTestContract,
    VerifierContract,
)
from rl_task_foundry.synthesis.registration_runner import (
    GeneratedArtifactBundle,
    RegistrationArtifactName,
    RegistrationBundleReport,
    RegistrationBundleStatus,
    VerifierProbeSpec,
    run_registration_bundle,
)
from rl_task_foundry.synthesis.subprocess_pool import RegistrationSubprocessPool

CURRENT_SYNTHESIS_GENERATOR_VERSION = "milestone-3-runtime-v1"
RUNTIME_OWNED_ENVIRONMENT_FIELDS = frozenset(
    {
        "env_id",
        "db_id",
        "domain",
        "category",
        "difficulty_vector",
        "created_at",
        "generator_version",
        "tool_signature",
        "task_signature",
        "verifier_signature",
        "status",
        "quality_metrics",
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
    tools: list[ToolContract] = Field(default_factory=list)
    task: TaskContract
    solution: SolutionContract
    tool_self_test: ToolSelfTestContract = Field(default_factory=ToolSelfTestContract)
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
    domain_name: str
    user_role: str
    agent_role: str
    scenario_description: str
    requested_category: CategoryTaxonomy | None = None
    schema_summary: dict[str, object] = Field(default_factory=dict)
    previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput] = Field(default_factory=dict)
    memory: list[SynthesisMemoryEntry] = Field(default_factory=list)


class SynthesisStageResult(StrictModel):
    phase: SynthesisPhase
    provider: str
    model: str
    payload: SynthesisPhaseOutput
    memory_entry: SynthesisMemoryEntry
    tool_traces: list[SynthesisToolTraceEntry] = Field(default_factory=list)


class SynthesisEnvironmentDraft(StrictModel):
    created_at: datetime
    db_id: str
    requested_category: CategoryTaxonomy
    schema_summary: dict[str, object] = Field(default_factory=dict)
    selected_category: CategoryTaxonomy
    environment: EnvironmentContract
    artifacts: GeneratedArtifactBundle
    registration_report: RegistrationBundleReport
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


class SynthesisPhaseExecutionError(SynthesisRuntimeError):
    """Raised when a synthesis phase fails across all candidate backends."""


class SynthesisProviderUnavailableError(SynthesisRuntimeError):
    """Raised when all provider candidates are currently unavailable."""


class SynthesisCategoryMismatchError(SynthesisRuntimeError):
    """Raised when category inference diverges from the requested category."""


class SynthesisRegistrationError(SynthesisRuntimeError):
    """Raised when generated artifacts fail registration validation."""


class SynthesisDbBindingError(SynthesisRuntimeError):
    """Raised when a runtime instance is reused for a different logical database id."""


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
    _bound_db_id: str | None = field(default=None, init=False, repr=False)
    _bind_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _graph_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _registration_lock: asyncio.Lock = field(
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

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
        graph: SchemaGraph | None = None,
    ) -> SynthesisEnvironmentDraft:
        await self._bind_db_id(db_id)
        resolved_graph = graph if graph is not None else await self._introspect_graph()
        schema_summary = summarize_schema_graph(resolved_graph)

        stage_results: list[SynthesisStageResult] = []
        memory: list[SynthesisMemoryEntry] = []
        tool_traces: list[SynthesisToolTraceEntry] = []
        previous_outputs: dict[SynthesisPhase, SynthesisPhaseOutput] = {}

        for phase in (
            SynthesisPhase.SCHEMA_EXPLORATION,
            SynthesisPhase.CATEGORY_INFERENCE,
            SynthesisPhase.ARTIFACT_GENERATION,
        ):
            request = SynthesisStageRequest(
                phase=phase,
                db_id=db_id,
                domain_name=self.config.domain.name,
                user_role=self.config.domain.user_role,
                agent_role=self.config.domain.agent_role,
                scenario_description=self.config.domain.scenario_description,
                requested_category=requested_category,
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

        artifact_payload = previous_outputs[SynthesisPhase.ARTIFACT_GENERATION]
        assert isinstance(artifact_payload, ArtifactGenerationOutput)
        if artifact_payload.proposed_environment.task.category != requested_category:
            raise SynthesisCategoryMismatchError(
                "artifact generation returned a task with the wrong category"
            )
        registration_report = await self._execute_registration_gate(
            artifact_payload.artifacts,
            proposed_environment=artifact_payload.proposed_environment,
        )

        materialized_at = datetime.now(timezone.utc)
        environment = self._materialize_environment(
            proposed_environment=artifact_payload.proposed_environment,
            artifacts=artifact_payload.artifacts,
            db_id=db_id,
            requested_category=requested_category,
            created_at=materialized_at,
        )

        return SynthesisEnvironmentDraft(
            created_at=materialized_at,
            db_id=db_id,
            requested_category=requested_category,
            schema_summary=schema_summary,
            selected_category=selected_category,
            environment=environment,
            artifacts=artifact_payload.artifacts,
            registration_report=registration_report,
            stage_results=stage_results,
            memory=memory,
            tool_traces=tool_traces,
            provider_status=self.provider_status(),
        )

    async def close(self) -> None:
        if self._registration_pool is not None:
            await self._registration_pool.close()
            self._registration_pool = None

    def provider_status(self) -> dict[str, SynthesisProviderStatus]:
        return {
            provider_name: _snapshot_to_status(breaker.snapshot())
            for provider_name, breaker in self._breakers.items()
        }

    async def _run_phase(self, request: SynthesisStageRequest) -> SynthesisStageResult:
        candidate_backends = self.phase_backends.get(request.phase, []) if self.phase_backends else []
        if not candidate_backends:
            raise SynthesisPhaseExecutionError(
                f"no synthesis backends configured for {request.phase.value}"
            )

        errors: list[str] = []
        for backend in candidate_backends:
            breaker = self._breakers[backend.provider_name]
            if not breaker.is_available():
                continue
            try:
                result = await backend.run_stage(request)
            except Exception as exc:  # pragma: no cover
                breaker.record_failure()
                errors.append(f"{backend.provider_name}/{backend.model_name}: {type(exc).__name__}")
                continue
            breaker.record_success()
            return result

        if errors:
            raise SynthesisPhaseExecutionError(
                f"synthesis phase {request.phase.value} failed across candidate providers: "
                + ", ".join(errors)
            )
        raise SynthesisProviderUnavailableError(
            f"all providers are in cooldown for synthesis phase {request.phase.value}"
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

    async def _execute_registration_gate(
        self,
        bundle: GeneratedArtifactBundle,
        *,
        proposed_environment: ProposedEnvironmentDraft,
    ) -> RegistrationBundleReport:
        try:
            report = await self._run_registration_gate(
                bundle=bundle,
                proposed_environment=proposed_environment,
            )
        except Exception as exc:
            raise SynthesisRegistrationError("registration gate execution failed") from exc
        if report.status != RegistrationBundleStatus.PASSED:
            raise SynthesisRegistrationError("generated artifacts failed registration validation")
        return report

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
            pool=self._registration_pool,
            verifier_probe_specs=self._build_verifier_probe_specs(proposed_environment),
        )

    def _materialize_environment(
        self,
        *,
        proposed_environment: ProposedEnvironmentDraft,
        artifacts: GeneratedArtifactBundle,
        db_id: str,
        requested_category: CategoryTaxonomy,
        created_at: datetime,
    ) -> EnvironmentContract:
        task_payload = proposed_environment.task.model_dump(mode="python")
        task_signature = self._signature_for_payload(task_payload)
        tool_signature = self._signature_for_text(
            artifacts.tool_source + "\n---self-test---\n" + artifacts.tool_self_test_source
        )
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
                "difficulty_vector": proposed_environment.task.difficulty_vector,
                "created_at": created_at,
                "generator_version": CURRENT_SYNTHESIS_GENERATOR_VERSION,
                "tool_signature": tool_signature,
                "task_signature": task_signature,
                "verifier_signature": verifier_signature,
                "status": EnvironmentStatus.DRAFT,
                "quality_metrics": EnvironmentQualityMetrics().model_dump(mode="python"),
                "task": proposed_environment.task.model_copy(
                    update={"category": requested_category}
                ).model_dump(mode="python"),
            }
        )
        return EnvironmentContract.model_validate(payload)

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
