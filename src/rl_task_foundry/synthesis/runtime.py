"""Synthesis meta-agent runtime skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
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
    EnvironmentContract,
    StrictModel,
)
from rl_task_foundry.synthesis.registration_runner import GeneratedArtifactBundle


class SynthesisPhase(StrEnum):
    SCHEMA_EXPLORATION = "schema_exploration"
    CATEGORY_INFERENCE = "category_inference"
    ARTIFACT_GENERATION = "artifact_generation"


class SynthesisProviderStatus(StrictModel):
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


class SynthesisStageRequest(StrictModel):
    phase: SynthesisPhase
    db_id: str
    domain_name: str
    user_role: str
    agent_role: str
    scenario_description: str
    requested_category: CategoryTaxonomy | None = None
    schema_summary: dict[str, object] = Field(default_factory=dict)
    previous_outputs: dict[str, dict[str, object]] = Field(default_factory=dict)
    memory: list[SynthesisMemoryEntry] = Field(default_factory=list)


class SynthesisStageResult(StrictModel):
    phase: SynthesisPhase
    provider: str
    model: str
    payload: dict[str, object] = Field(default_factory=dict)
    memory_entry: SynthesisMemoryEntry
    tool_traces: list[SynthesisToolTraceEntry] = Field(default_factory=list)


class SynthesisEnvironmentDraft(StrictModel):
    db_id: str
    requested_category: CategoryTaxonomy
    schema_summary: dict[str, object] = Field(default_factory=dict)
    selected_category: CategoryTaxonomy
    environment: EnvironmentContract
    artifacts: GeneratedArtifactBundle
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


def summarize_schema_graph(graph: SchemaGraph) -> dict[str, object]:
    table_summaries: list[dict[str, object]] = []
    for table in graph.tables:
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
        "tables": table_summaries,
    }


def _snapshot_to_status(snapshot: ProviderCircuitSnapshot) -> SynthesisProviderStatus:
    return SynthesisProviderStatus(
        total_requests=snapshot.total_requests,
        failures=snapshot.failures,
        error_rate=snapshot.error_rate,
        available=snapshot.available,
        cooldown_remaining_s=snapshot.cooldown_remaining_s,
    )


@dataclass(slots=True)
class SynthesisAgentRuntime:
    config: AppConfig
    phase_backends: dict[SynthesisPhase, list[SynthesisStageBackend]] | None = None
    _breakers: dict[str, ProviderCircuitBreaker] = field(default_factory=dict, init=False, repr=False)

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
        resolved_graph = graph if graph is not None else await self._introspect_graph()
        schema_summary = summarize_schema_graph(resolved_graph)

        stage_results: list[SynthesisStageResult] = []
        memory: list[SynthesisMemoryEntry] = []
        tool_traces: list[SynthesisToolTraceEntry] = []
        previous_outputs: dict[str, dict[str, object]] = {}

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
            previous_outputs[phase.value] = result.payload

        category_payload = previous_outputs[SynthesisPhase.CATEGORY_INFERENCE.value]
        selected_category = CategoryTaxonomy(category_payload["selected_category"])
        if selected_category != requested_category:
            raise SynthesisCategoryMismatchError(
                "category inference result did not match the requested category"
            )

        artifact_payload = previous_outputs[SynthesisPhase.ARTIFACT_GENERATION.value]
        environment = EnvironmentContract.model_validate(artifact_payload["environment"])
        artifacts = GeneratedArtifactBundle.model_validate(artifact_payload["artifacts"])
        if environment.category != requested_category:
            raise SynthesisCategoryMismatchError(
                "generated environment category did not match the requested category"
            )

        return SynthesisEnvironmentDraft(
            db_id=db_id,
            requested_category=requested_category,
            schema_summary=schema_summary,
            selected_category=selected_category,
            environment=environment,
            artifacts=artifacts,
            stage_results=stage_results,
            memory=memory,
            tool_traces=tool_traces,
            provider_status=self.provider_status(),
        )

    def provider_status(self) -> dict[str, SynthesisProviderStatus]:
        return {
            provider_name: _snapshot_to_status(breaker.snapshot())
            for provider_name, breaker in self._breakers.items()
        }

    async def _run_phase(self, request: SynthesisStageRequest) -> SynthesisStageResult:
        candidate_backends = self.phase_backends.get(request.phase, []) if self.phase_backends else []
        if not candidate_backends:
            raise SynthesisPhaseExecutionError(f"no synthesis backends configured for {request.phase.value}")

        errors: list[str] = []
        for backend in candidate_backends:
            breaker = self._breakers[backend.provider_name]
            if not breaker.is_available():
                continue
            try:
                result = await backend.run_stage(request)
            except Exception as exc:  # pragma: no cover - exercised through tests with fake backends
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
        introspector = PostgresSchemaIntrospector(
            database=self.config.database,
            default_visibility=self.config.privacy.default_visibility,
            visibility_overrides=self.config.privacy.visibility_overrides,
        )
        return await introspector.introspect()
