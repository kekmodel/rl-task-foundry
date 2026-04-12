from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import ModelRef, OutputConfig, ProviderConfig
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle, AtomicToolGenerator
from rl_task_foundry.synthesis import backend_openai_agents as backend_module
from rl_task_foundry.synthesis.backend_openai_agents import OpenAIAgentsSynthesisBackend
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    ConstraintKind,
    ConstraintSummaryItem,
    DifficultyAxis,
    EnvironmentContract,
    EnvironmentStatus,
    MaterializedFactsSchema,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    ShadowVerifierContract,
    SolutionContract,
    TaskContract,
    VerifierContract,
)
from rl_task_foundry.synthesis.registration_policy import ArtifactKind
from rl_task_foundry.synthesis.registration_runner import (
    ArtifactRegistrationResult,
    GeneratedArtifactBundle,
    RegistrationArtifactName,
    RegistrationBundleReport,
    RegistrationBundleStatus,
)
from rl_task_foundry.synthesis.registration_policy import RegistrationError, VerifierHybridAnalysis
from rl_task_foundry.synthesis.runtime import (
    ArtifactGenerationOutput,
    CURRENT_SYNTHESIS_GENERATOR_VERSION,
    CategoryInferenceOutput,
    ProposedEnvironmentDraft,
    RUNTIME_OWNED_ENVIRONMENT_FIELDS,
    SchemaExplorationOutput,
    SynthesisAgentRuntime,
    SynthesisCategoryBackoffError,
    SynthesisCategoryMismatchError,
    SynthesisDbBindingError,
    SynthesisMemoryEntry,
    SynthesisPhase,
    SynthesisRegistrationError,
    SynthesisSelfConsistencyError,
    SynthesisSelfConsistencyDiagnostics,
    SynthesisSelfConsistencyOutcome,
    SynthesisStageRequest,
    SynthesisStageResult,
    SynthesisToolTraceEntry,
)
from rl_task_foundry.synthesis.scheduler import (
    SynthesisDbSnapshot,
    SynthesisDomainScheduler,
    SynthesisSelectionStatus,
)
from rl_task_foundry.synthesis.subprocess_pool import (
    RegistrationSelfConsistencyResult,
    RegistrationVerifierProbeResult,
)


def _sample_graph() -> SchemaGraph:
    customer_id = ColumnProfile(
        schema_name="public",
        table_name="customer",
        column_name="customer_id",
        data_type="integer",
        ordinal_position=1,
        is_nullable=False,
        visibility="user_visible",
        is_primary_key=True,
    )
    customer_name = ColumnProfile(
        schema_name="public",
        table_name="customer",
        column_name="name",
        data_type="text",
        ordinal_position=2,
        is_nullable=False,
        visibility="user_visible",
    )
    order_id = ColumnProfile(
        schema_name="public",
        table_name="orders",
        column_name="order_id",
        data_type="integer",
        ordinal_position=1,
        is_nullable=False,
        visibility="internal",
        is_primary_key=True,
    )
    order_customer_id = ColumnProfile(
        schema_name="public",
        table_name="orders",
        column_name="customer_id",
        data_type="integer",
        ordinal_position=2,
        is_nullable=False,
        visibility="internal",
        is_foreign_key=True,
    )
    return SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="customer",
                columns=[customer_id, customer_name],
                primary_key=("customer_id",),
                row_estimate=5,
            ),
            TableProfile(
                schema_name="public",
                table_name="orders",
                columns=[order_id, order_customer_id],
                primary_key=("order_id",),
                row_estimate=20,
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_customer_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("customer_id",),
                target_schema="public",
                target_table="customer",
                target_columns=("customer_id",),
                source_is_unique=False,
                fanout_estimate=4.0,
            )
        ],
    )


def _config_with_synthesis_output(tmp_path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        accepted_jsonl_path=tmp_path / "accepted.jsonl",
        rejected_jsonl_path=tmp_path / "rejected.jsonl",
        events_jsonl_path=tmp_path / "events.jsonl",
        traces_dir=tmp_path / "traces",
    )
    return config.model_copy(update={"output": output}, deep=True)


def _sample_proposed_environment(
    category: CategoryTaxonomy = CategoryTaxonomy.ASSIGNMENT,
    difficulty_vector: dict[DifficultyAxis, float] | None = None,
) -> ProposedEnvironmentDraft:
    output_schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="assignments",
            type=OutputFieldType.LIST,
            ordered=True,
            items=OutputFieldContract(
                name="assignment",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="day", type=OutputFieldType.DATE),
                    OutputFieldContract(name="customer", type=OutputFieldType.STRING),
                ],
            ),
        ),
        primary_output_format="json_array",
    )
    return ProposedEnvironmentDraft(
        task=TaskContract(
            question="고객 배정 계획을 만들어 주세요.",
            category=category,
            output_schema=output_schema,
            constraint_summary=[
                ConstraintSummaryItem(
                    key="unique_customer",
                    kind=ConstraintKind.UNIQUENESS,
                    summary="같은 고객을 중복 배정하지 않는다.",
                )
            ],
            difficulty_vector=difficulty_vector or {},
        ),
        solution=SolutionContract(),
        verifier=VerifierContract(facts_schema=MaterializedFactsSchema()),
        shadow_verifier=ShadowVerifierContract(facts_schema=MaterializedFactsSchema()),
        instance_space={
            "anchor_query": {
                "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                "outputs": ["customer_id"],
            }
        },
    )


def _sample_artifacts() -> GeneratedArtifactBundle:
    return GeneratedArtifactBundle(
        solution_source="def solve(tools):\n    return {'assignments': []}\n",
        verifier_source=(
            "async def fetch_facts(answer, tools):\n    return {}\n\n"
            "def facts_match_answer_claims(answer, facts):\n    return True\n\n"
            "def check_constraints(answer, facts):\n    return True\n\n"
            "def verify(answer, tools):\n    return True\n"
        ),
        shadow_verifier_source=(
            "async def fetch_facts(answer, tools):\n    return {}\n\n"
            "def facts_match_answer_claims(answer, facts):\n    return True\n\n"
            "def check_constraints(answer, facts):\n    return True\n\n"
            "def verify(answer, tools):\n    return True\n"
        ),
    )


def _sample_atomic_tool_bundle(db_id: str = "sakila") -> AtomicToolBundle:
    return AtomicToolBundle(
        db_id=db_id,
        tools=[],
        source="async def lookup_city(conn, customer_id):\n    return {'city': 'sasebo'}\n",
    )


def _payloads(
    *,
    category: CategoryTaxonomy = CategoryTaxonomy.ASSIGNMENT,
) -> dict[SynthesisPhase, object]:
    return {
        SynthesisPhase.SCHEMA_EXPLORATION: SchemaExplorationOutput(
            domain_hypothesis="customer_support",
            candidate_categories=[category],
            memory_summary="schema explored",
        ),
        SynthesisPhase.CATEGORY_INFERENCE: CategoryInferenceOutput(
            selected_category=category,
            rationale="best match",
            memory_summary="category selected",
        ),
        SynthesisPhase.ARTIFACT_GENERATION: ArtifactGenerationOutput(
            proposed_environment=_sample_proposed_environment(category),
            artifacts=_sample_artifacts(),
            memory_summary="artifacts generated",
        ),
    }


def _passing_registration_report() -> RegistrationBundleReport:
    def _artifact_result(
        name: RegistrationArtifactName, kind: ArtifactKind
    ) -> ArtifactRegistrationResult:
        executed = name == RegistrationArtifactName.TOOL_SELF_TEST
        return ArtifactRegistrationResult(
            artifact_name=name,
            artifact_kind=kind,
            execution_required=executed,
            executed=executed,
            execution_call_count=1 if executed else None,
            execution_return_value={"ok": True} if executed else None,
        )

    return RegistrationBundleReport(
        status=RegistrationBundleStatus.PASSED,
        tool=_artifact_result(RegistrationArtifactName.TOOL, ArtifactKind.TOOL_MODULE),
        tool_self_test=_artifact_result(
            RegistrationArtifactName.TOOL_SELF_TEST,
            ArtifactKind.TOOL_SELF_TEST_MODULE,
        ),
        solution=_artifact_result(RegistrationArtifactName.SOLUTION, ArtifactKind.SOLUTION_MODULE),
        verifier=_artifact_result(RegistrationArtifactName.VERIFIER, ArtifactKind.VERIFIER_MODULE),
        shadow_verifier=_artifact_result(
            RegistrationArtifactName.SHADOW_VERIFIER,
            ArtifactKind.SHADOW_VERIFIER_MODULE,
        ),
    )

def _diagnostic_registration_report(
    *,
    status: RegistrationBundleStatus = RegistrationBundleStatus.PASSED,
    probe_error_codes: list[str] | None = None,
) -> RegistrationBundleReport:
    probe_errors = [
        RegistrationError(code=code, detail=f"diagnostic error: {code}")
        for code in (probe_error_codes or [])
    ]
    verifier = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.VERIFIER,
        artifact_kind=ArtifactKind.VERIFIER_MODULE,
        probe_required=True,
        probe_executed=True,
        probe_errors=probe_errors,
        verifier_hybrid_analysis=VerifierHybridAnalysis(
            fetch_facts_tool_calls=1,
            fetch_facts_answer_references=1,
            facts_match_answer_references=1,
            facts_match_facts_references=1,
            check_constraints_facts_references=1,
        ),
        verifier_execution_probe=RegistrationVerifierProbeResult(
            request_id="probe-1",
            worker_pid=1234,
            errors=probe_errors,
            fetch_facts_return_keys=["city"],
            expected_fact_keys=["city"],
            fetch_facts_tool_calls=1,
            verify_tool_calls=1,
            facts_match_result=True,
            check_constraints_result=True,
            verify_result=True,
        ),
    )
    return _passing_registration_report().model_copy(
        update={
            "status": status,
            "verifier": verifier,
            "shadow_verifier": verifier.model_copy(
                update={"artifact_name": RegistrationArtifactName.SHADOW_VERIFIER}
            ),
        }
    )


def _passing_self_consistency_result() -> RegistrationSelfConsistencyResult:
    return RegistrationSelfConsistencyResult(
        request_id="self-check-1",
        worker_pid=1234,
        errors=[],
        answer={"assignments": []},
        solution_tool_calls=1,
        verifier_tool_calls=1,
        shadow_verifier_tool_calls=1,
        fetch_facts_return_keys=[],
        expected_fact_keys=[],
        fetch_facts_answer_reads=1,
        facts_match_answer_reads=1,
        facts_match_facts_reads=1,
        check_constraints_facts_reads=1,
        facts_match_result=True,
        check_constraints_result=True,
        verify_result=True,
        shadow_verify_result=True,
    )


def _failing_self_consistency_result(
    *,
    error_codes: list[str] | None = None,
    solution_tool_calls: int | None = 1,
    verifier_tool_calls: int | None = 1,
    shadow_verifier_tool_calls: int | None = 1,
    fetch_facts_answer_reads: int | None = 1,
    facts_match_answer_reads: int | None = 1,
    facts_match_facts_reads: int | None = 1,
    check_constraints_facts_reads: int | None = 1,
    facts_match_result: bool | None = True,
    check_constraints_result: bool | None = False,
    verify_result: bool | None = False,
    shadow_verify_result: bool | None = False,
    answer: object | None = None,
) -> RegistrationSelfConsistencyResult:
    return RegistrationSelfConsistencyResult(
        request_id="self-check-1",
        worker_pid=1234,
        errors=[
            RegistrationError(code=code, detail=f"self-consistency error: {code}")
            for code in (error_codes or [])
        ],
        answer={"assignments": []} if answer is None else answer,
        solution_tool_calls=solution_tool_calls,
        verifier_tool_calls=verifier_tool_calls,
        shadow_verifier_tool_calls=shadow_verifier_tool_calls,
        fetch_facts_return_keys=["city"],
        expected_fact_keys=["city"],
        fetch_facts_answer_reads=fetch_facts_answer_reads,
        facts_match_answer_reads=facts_match_answer_reads,
        facts_match_facts_reads=facts_match_facts_reads,
        check_constraints_facts_reads=check_constraints_facts_reads,
        facts_match_result=facts_match_result,
        check_constraints_result=check_constraints_result,
        verify_result=verify_result,
        shadow_verify_result=shadow_verify_result,
    )


class _FakeBackend:
    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        payloads: dict[SynthesisPhase, object],
        payload_repair_codes_by_phase: dict[SynthesisPhase, list[str]] | None = None,
        fail_phases: set[SynthesisPhase] | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
        self.payloads = payloads
        self.payload_repair_codes_by_phase = payload_repair_codes_by_phase or {}
        self.fail_phases = fail_phases or set()
        self.calls: list[SynthesisPhase] = []
        self.requests: list[SynthesisStageRequest] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def run_stage(self, request: SynthesisStageRequest) -> SynthesisStageResult:
        self.calls.append(request.phase)
        self.requests.append(request)
        if request.phase in self.fail_phases:
            raise RuntimeError(f"{self.provider_name} failed")
        payload = self.payloads[request.phase]
        return SynthesisStageResult(
            phase=request.phase,
            provider=self.provider_name,
            model=self.model_name,
            payload=payload,
            payload_repair_codes=self.payload_repair_codes_by_phase.get(request.phase, []),
            memory_entry=SynthesisMemoryEntry(
                phase=request.phase,
                provider=self.provider_name,
                model=self.model_name,
                summary=getattr(payload, "memory_summary", f"{request.phase.value} done"),
                turn_count=1,
                token_usage={"requests": 1},
            ),
            tool_traces=[
                SynthesisToolTraceEntry(
                    phase=request.phase,
                    provider=self.provider_name,
                    model=self.model_name,
                    tool_name=f"{request.phase.value}_tool",
                )
            ],
        )


class _AttemptAwareBackend(_FakeBackend):
    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        base_payloads: dict[SynthesisPhase, object],
        artifact_payloads_by_attempt: dict[int, ArtifactGenerationOutput],
    ) -> None:
        super().__init__(
            provider_name=provider_name,
            model_name=model_name,
            payloads=base_payloads,
        )
        self.artifact_payloads_by_attempt = artifact_payloads_by_attempt

    async def run_stage(self, request: SynthesisStageRequest) -> SynthesisStageResult:
        if request.phase == SynthesisPhase.ARTIFACT_GENERATION:
            payload = self.artifact_payloads_by_attempt[request.attempt_index]
            self.payloads = {**self.payloads, request.phase: payload}
        return await super().run_stage(request)


def test_proposed_and_materialized_environment_schemas_align() -> None:
    proposed_fields = set(ProposedEnvironmentDraft.model_fields)
    materialized_fields = set(EnvironmentContract.model_fields)

    assert proposed_fields == materialized_fields - RUNTIME_OWNED_ENVIRONMENT_FIELDS


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_builds_environment_draft_and_rewrites_trust_fields(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml")
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        assert bundle == _sample_artifacts()
        assert proposed_environment == _sample_proposed_environment()
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert draft.selected_category == CategoryTaxonomy.ASSIGNMENT
    assert draft.environment.status == EnvironmentStatus.DRAFT
    assert draft.environment.db_id == "sakila"
    assert draft.environment.domain == config.domain.name
    assert draft.environment.atomic_tool_set_ref == "db://sakila"
    assert draft.environment.generator_version == CURRENT_SYNTHESIS_GENERATOR_VERSION
    assert draft.environment.env_id.startswith("env_assignment_")
    assert len(draft.environment.env_id.split("_")[-1]) == 16
    assert draft.atomic_tool_bundle.db_id == "sakila"
    assert draft.atomic_tool_bundle.tools
    assert draft.environment.quality_metrics.self_consistency_pass is True
    assert draft.environment.tool_signature.startswith("sha256:")
    assert draft.environment.task_signature.startswith("sha256:")
    assert draft.environment.verifier_signature.startswith("sha256:")
    assert draft.environment.rollout_constraints == RolloutConstraintsContract(
        max_turns=config.solver_runtime.max_turns,
        max_episode_duration_ms=(
            config.database.statement_timeout_ms * config.solver_runtime.max_turns
        ),
        max_tool_rows=config.atomic_tools.bounded_result_limit,
    )
    assert draft.registration_report.status == RegistrationBundleStatus.PASSED
    assert draft.registration_diagnostics.status == RegistrationBundleStatus.PASSED
    assert draft.self_consistency_diagnostics.passed is True
    assert draft.self_consistency_diagnostics.shadow_verify_result is True
    assert draft.registration_diagnostics.error_codes == []
    assert draft.registration_diagnostics.verifier.fetch_facts_tool_calls == 1
    assert draft.registration_diagnostics.verifier.probe_fetch_facts_return_keys == ["city"]
    assert len(draft.instances) == 1
    assert len(draft.canonical_answers) == 1
    assert draft.instances[0].instance_id == "instance_0001"
    assert "Submit Result Format:" in draft.instances[0].rendered_user_prompt
    assert draft.canonical_answers[0].canonical_answer == []
    assert draft.canonical_answers[0].canonical_answer_json == canonical_json([])
    assert draft.environment.cross_instance_set.minimum_required == 1
    assert (
        draft.environment.cross_instance_set.instances[0].expected_solution_fingerprint
        == draft.canonical_answers[0].solution_fingerprint
    )
    assert [entry.phase for entry in draft.memory] == list(SynthesisPhase)
    assert draft.provider_status["codex_oauth"].observed_at.tzinfo is not None
    for request in backend.requests:
        assert request.atomic_tool_set_ref == "db://sakila"
        assert request.available_atomic_tools
        assert all("name" in tool for tool in request.available_atomic_tools)


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_records_payload_repair_codes_in_diagnostics(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml")
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
        payload_repair_codes_by_phase={
            SynthesisPhase.ARTIFACT_GENERATION: ["artifact_key_remapped"]
        },
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert draft.self_consistency_diagnostics.payload_repair_codes == [
        "artifact_key_remapped"
    ]


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_canonicalizes_materialized_answers(monkeypatch):
    config = load_config("rl_task_foundry.yaml")
    proposed_environment = ProposedEnvironmentDraft(
        task=TaskContract(
            question="일정을 시간순으로 정리해 주세요.",
            category=CategoryTaxonomy.ITINERARY,
            output_schema=OutputSchemaContract(
                root=OutputFieldContract(
                    name="itinerary",
                    type=OutputFieldType.LIST,
                    ordered=False,
                    sort_key=("time",),
                    items=OutputFieldContract(
                        name="entry",
                        type=OutputFieldType.OBJECT,
                        fields=[
                            OutputFieldContract(name="time", type=OutputFieldType.DATE),
                            OutputFieldContract(name="city", type=OutputFieldType.STRING),
                        ],
                    ),
                ),
                primary_output_format="json_array",
            ),
            difficulty_vector={},
        ),
        solution=SolutionContract(),
        verifier=VerifierContract(facts_schema=MaterializedFactsSchema()),
        shadow_verifier=ShadowVerifierContract(facts_schema=MaterializedFactsSchema()),
        instance_space={
            "anchor_query": {
                "sql": "SELECT city FROM itinerary ORDER BY day",
                "outputs": ["city"],
            }
        },
    )
    payloads = _payloads(category=CategoryTaxonomy.ITINERARY)
    payloads[SynthesisPhase.ARTIFACT_GENERATION] = ArtifactGenerationOutput(
        proposed_environment=proposed_environment,
        artifacts=_sample_artifacts(),
        memory_summary="artifacts generated",
    )
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=payloads,
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _passing_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _failing_self_consistency_result(
            error_codes=[],
            verify_result=True,
            check_constraints_result=True,
            shadow_verify_result=True,
            answer=[
                {"time": "2026-10-03", "city": "Busan"},
                {"time": "2026-10-01", "city": "Seoul"},
                {"time": "2026-10-02", "city": "Jeju"},
            ],
        )

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ITINERARY,
        graph=_sample_graph(),
    )

    assert draft.canonical_answers[0].canonical_answer == [
        {"time": "2026-10-01", "city": "Seoul"},
        {"time": "2026-10-02", "city": "Jeju"},
        {"time": "2026-10-03", "city": "Busan"},
    ]


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_materializes_atomic_tool_bundle_once(
    tmp_path,
    monkeypatch,
):
    config = _config_with_synthesis_output(tmp_path)
    payloads = _payloads()
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=payloads,
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )
    generate_calls: list[str] = []
    original_generate_bundle = AtomicToolGenerator.generate_bundle

    def _wrapped_generate_bundle(self, graph, *, db_id: str):
        generate_calls.append(db_id)
        return original_generate_bundle(self, graph, db_id=db_id)

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    monkeypatch.setattr(AtomicToolGenerator, "generate_bundle", _wrapped_generate_bundle)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    first = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )
    second = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    bundle_dir = tmp_path / "databases" / "sakila"
    assert generate_calls == ["sakila"]
    assert bundle_dir.joinpath("atomic_tools.py").exists()
    assert bundle_dir.joinpath("atomic_tool_definitions.json").exists()
    assert first.atomic_tool_bundle.source == second.atomic_tool_bundle.source
    assert (
        bundle_dir.joinpath("atomic_tools.py").read_text(encoding="utf-8")
        == first.atomic_tool_bundle.source
    )


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_reuses_provider_resilience_for_fallback(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml")
    failing = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
        fail_phases={SynthesisPhase.SCHEMA_EXPLORATION, SynthesisPhase.CATEGORY_INFERENCE},
    )
    fallback = _FakeBackend(
        provider_name="local_server",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [failing, fallback] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _passing_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert draft.environment.env_id.startswith("env_assignment_")
    assert failing.calls == [
        SynthesisPhase.SCHEMA_EXPLORATION,
        SynthesisPhase.CATEGORY_INFERENCE,
    ]
    assert fallback.calls == list(SynthesisPhase)
    assert runtime.provider_status()["codex_oauth"].available is False
    assert runtime.provider_status()["codex_oauth"].failures == 2


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_rejects_category_mismatch(monkeypatch):
    config = load_config("rl_task_foundry.yaml")
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(category=CategoryTaxonomy.OTHER),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _passing_registration_report()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

    with pytest.raises(SynthesisCategoryMismatchError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_rejects_wrong_task_category_in_artifact_generation(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 1
    mismatched = _sample_proposed_environment(CategoryTaxonomy.OTHER)
    payloads = _payloads()
    payloads[SynthesisPhase.ARTIFACT_GENERATION] = ArtifactGenerationOutput(
        proposed_environment=mismatched,
        artifacts=_sample_artifacts(),
        memory_summary="artifacts generated",
    )
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=payloads,
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _passing_registration_report()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

    with pytest.raises(SynthesisSelfConsistencyError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )
    assert [attempt.outcome for attempt in exc_info.value.attempts] == [
        SynthesisSelfConsistencyOutcome.CATEGORY_MISMATCH
    ]


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_rejects_failed_registration(monkeypatch):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 1
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report(
            status=RegistrationBundleStatus.FAILED,
            probe_error_codes=["facts_schema_keys_mismatch"],
        )

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    with pytest.raises(SynthesisSelfConsistencyError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )
    assert exc_info.value.last_registration_diagnostics is not None
    assert exc_info.value.last_registration_diagnostics.error_codes == [
        "facts_schema_keys_mismatch"
    ]
    assert exc_info.value.last_registration_diagnostics.failing_artifacts == [
        RegistrationArtifactName.VERIFIER,
        RegistrationArtifactName.SHADOW_VERIFIER,
    ]


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_retries_artifact_generation_with_registration_feedback(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml")
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )
    reports = iter(
        [
            _diagnostic_registration_report(
                status=RegistrationBundleStatus.FAILED,
                probe_error_codes=["facts_schema_keys_mismatch"],
            ),
            _diagnostic_registration_report(),
        ]
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return next(reports)

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    artifact_requests = [
        request for request in backend.requests if request.phase == SynthesisPhase.ARTIFACT_GENERATION
    ]
    assert [request.attempt_index for request in artifact_requests] == [1, 2]
    assert artifact_requests[0].latest_registration_diagnostics is None
    assert artifact_requests[1].latest_registration_diagnostics is not None
    assert artifact_requests[1].latest_registration_diagnostics.error_codes == [
        "facts_schema_keys_mismatch"
    ]
    assert [attempt.outcome for attempt in draft.self_consistency_attempts] == [
        SynthesisSelfConsistencyOutcome.REGISTRATION_FAILED,
        SynthesisSelfConsistencyOutcome.PASSED,
    ]
    assert (
        draft.self_consistency_attempts[0].registration_diagnostics.error_codes
        == ["facts_schema_keys_mismatch"]
    )
    assert backend.calls.count(SynthesisPhase.SCHEMA_EXPLORATION) == 1
    assert backend.calls.count(SynthesisPhase.CATEGORY_INFERENCE) == 1
    assert backend.calls.count(SynthesisPhase.ARTIFACT_GENERATION) == 2


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_raises_self_consistency_error_after_budget_exhausted(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 2
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report(
            status=RegistrationBundleStatus.FAILED,
            probe_error_codes=["facts_schema_keys_mismatch"],
        )

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

    with pytest.raises(SynthesisSelfConsistencyError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    assert [attempt.outcome for attempt in exc_info.value.attempts] == [
        SynthesisSelfConsistencyOutcome.REGISTRATION_FAILED,
        SynthesisSelfConsistencyOutcome.REGISTRATION_FAILED,
    ]
    assert exc_info.value.last_registration_diagnostics is not None
    assert exc_info.value.last_registration_diagnostics.error_codes == [
        "facts_schema_keys_mismatch"
    ]
    assert backend.calls.count(SynthesisPhase.ARTIFACT_GENERATION) == 2


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_retries_on_self_consistency_failure(monkeypatch):
    config = load_config("rl_task_foundry.yaml")
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )
    checks = iter(
        [
            _failing_self_consistency_result(
                solution_tool_calls=0,
                facts_match_result=True,
                check_constraints_result=False,
                verify_result=False,
            ),
            _passing_self_consistency_result(),
        ]
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return next(checks)

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    artifact_requests = [
        request for request in backend.requests if request.phase == SynthesisPhase.ARTIFACT_GENERATION
    ]
    assert [request.attempt_index for request in artifact_requests] == [1, 2]
    assert artifact_requests[0].latest_self_consistency_diagnostics is None
    assert artifact_requests[1].latest_self_consistency_diagnostics is not None
    assert artifact_requests[1].latest_self_consistency_diagnostics.verify_result is False
    assert (
        "solution_missing_tool_usage"
        in artifact_requests[1].latest_self_consistency_diagnostics.weak_signal_codes
    )
    assert [attempt.outcome for attempt in draft.self_consistency_attempts] == [
        SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
        SynthesisSelfConsistencyOutcome.PASSED,
    ]
    assert draft.self_consistency_attempts[0].self_consistency_diagnostics is not None
    assert (
        draft.self_consistency_attempts[0].self_consistency_diagnostics.check_constraints_result
        is False
    )
    assert (
        "solution_missing_tool_usage"
        in draft.self_consistency_attempts[0].self_consistency_diagnostics.weak_signal_codes
    )
    assert draft.self_consistency_diagnostics.passed is True
    assert draft.environment.quality_metrics.self_consistency_pass is True


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_raises_after_self_consistency_budget_exhausted(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 2
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _failing_self_consistency_result(
            solution_tool_calls=0,
            facts_match_result=True,
            check_constraints_result=False,
            verify_result=False,
        )

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    with pytest.raises(SynthesisSelfConsistencyError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    assert [attempt.outcome for attempt in exc_info.value.attempts] == [
        SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
        SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
    ]
    assert exc_info.value.last_self_consistency_diagnostics is not None
    assert exc_info.value.last_self_consistency_diagnostics.verify_result is False
    assert backend.calls.count(SynthesisPhase.ARTIFACT_GENERATION) == 2


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_rejects_weakened_difficulty_vector_between_attempts(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 3
    payloads = _payloads()
    backend = _AttemptAwareBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        base_payloads=payloads,
        artifact_payloads_by_attempt={
            1: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={
                        DifficultyAxis.SLOT_COUNT: 3.0,
                        DifficultyAxis.CONSTRAINT_COUNT: 4.0,
                    }
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt one",
            ),
            2: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={DifficultyAxis.SLOT_COUNT: 2.0}
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt two weakened",
            ),
            3: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={
                        DifficultyAxis.SLOT_COUNT: 3.0,
                        DifficultyAxis.CONSTRAINT_COUNT: 5.0,
                    }
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt three strengthened",
            ),
        },
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )
    checks = iter(
        [
            _failing_self_consistency_result(
                solution_tool_calls=1,
                facts_match_result=True,
                check_constraints_result=False,
                verify_result=False,
            ),
            _passing_self_consistency_result(),
        ]
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return next(checks)

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert [attempt.outcome for attempt in draft.self_consistency_attempts] == [
        SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
        SynthesisSelfConsistencyOutcome.DIFFICULTY_WEAKENED,
        SynthesisSelfConsistencyOutcome.PASSED,
    ]
    assert "constraint_count" in draft.self_consistency_attempts[1].error_message
    artifact_requests = [
        request for request in backend.requests if request.phase == SynthesisPhase.ARTIFACT_GENERATION
    ]
    assert [request.attempt_index for request in artifact_requests] == [1, 2, 3]
    assert artifact_requests[2].latest_self_consistency_diagnostics is not None
    assert artifact_requests[2].latest_self_consistency_diagnostics.verify_result is False


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_rejects_multi_axis_difficulty_crank(monkeypatch):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 3
    payloads = _payloads()
    backend = _AttemptAwareBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        base_payloads=payloads,
        artifact_payloads_by_attempt={
            1: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={
                        DifficultyAxis.SLOT_COUNT: 3.0,
                        DifficultyAxis.CONSTRAINT_COUNT: 4.0,
                    }
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt one",
            ),
            2: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={
                        DifficultyAxis.SLOT_COUNT: 4.0,
                        DifficultyAxis.CONSTRAINT_COUNT: 5.0,
                    }
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt two invalid multi-axis crank",
            ),
            3: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={
                        DifficultyAxis.SLOT_COUNT: 3.0,
                        DifficultyAxis.CONSTRAINT_COUNT: 5.0,
                    }
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt three valid single-axis crank",
            ),
        },
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )
    registration_calls = 0
    checks = iter(
        [
            _failing_self_consistency_result(
                facts_match_result=True,
                check_constraints_result=False,
                verify_result=False,
            ),
            _passing_self_consistency_result(),
        ]
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        nonlocal registration_calls
        registration_calls += 1
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return next(checks)

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert [attempt.outcome for attempt in draft.self_consistency_attempts] == [
        SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
        SynthesisSelfConsistencyOutcome.DIFFICULTY_CRANK_INVALID,
        SynthesisSelfConsistencyOutcome.PASSED,
    ]
    assert draft.self_consistency_attempts[1].error_message is not None
    assert "constraint_count,slot_count" in draft.self_consistency_attempts[1].error_message
    assert registration_calls == 2
    artifact_requests = [
        request for request in backend.requests if request.phase == SynthesisPhase.ARTIFACT_GENERATION
    ]
    assert [request.attempt_index for request in artifact_requests] == [1, 2, 3]
    assert artifact_requests[0].difficulty_crank_index == 0
    assert artifact_requests[0].difficulty_crank_history == []
    assert artifact_requests[1].strongest_difficulty_vector == {
        DifficultyAxis.SLOT_COUNT: 3.0,
        DifficultyAxis.CONSTRAINT_COUNT: 4.0,
    }
    assert artifact_requests[1].difficulty_crank_index == 0
    assert artifact_requests[1].difficulty_crank_history == []
    assert artifact_requests[2].strongest_difficulty_vector == {
        DifficultyAxis.SLOT_COUNT: 3.0,
        DifficultyAxis.CONSTRAINT_COUNT: 4.0,
    }
    assert artifact_requests[2].difficulty_crank_index == 0
    assert artifact_requests[2].difficulty_crank_history == []
    assert artifact_requests[2].latest_self_consistency_diagnostics is not None


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_enforces_difficulty_crank_limit(monkeypatch):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 3
    config.synthesis.runtime.max_difficulty_cranks = 1
    payloads = _payloads()
    backend = _AttemptAwareBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        base_payloads=payloads,
        artifact_payloads_by_attempt={
            1: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={DifficultyAxis.SLOT_COUNT: 3.0}
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt one",
            ),
            2: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={DifficultyAxis.SLOT_COUNT: 4.0}
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt two valid crank",
            ),
            3: ArtifactGenerationOutput(
                proposed_environment=_sample_proposed_environment(
                    difficulty_vector={DifficultyAxis.SLOT_COUNT: 5.0}
                ),
                artifacts=_sample_artifacts(),
                memory_summary="attempt three crank limit exceeded",
            ),
        },
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )
    registration_calls = 0

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        nonlocal registration_calls
        registration_calls += 1
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _failing_self_consistency_result(
            facts_match_result=True,
            check_constraints_result=False,
            verify_result=False,
        )

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    with pytest.raises(SynthesisSelfConsistencyError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    assert [attempt.outcome for attempt in exc_info.value.attempts] == [
        SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
        SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED,
        SynthesisSelfConsistencyOutcome.DIFFICULTY_CRANK_LIMIT_EXCEEDED,
    ]
    assert exc_info.value.attempts[2].error_message == "difficulty crank limit exceeded: max=1"
    assert registration_calls == 2
    artifact_requests = [
        request for request in backend.requests if request.phase == SynthesisPhase.ARTIFACT_GENERATION
    ]
    assert [request.attempt_index for request in artifact_requests] == [1, 2, 3]
    assert artifact_requests[2].difficulty_crank_index == 1
    assert artifact_requests[2].difficulty_crank_history == [DifficultyAxis.SLOT_COUNT]
    assert artifact_requests[2].strongest_difficulty_vector == {
        DifficultyAxis.SLOT_COUNT: 4.0
    }


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_enters_category_backoff_after_consecutive_discards(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 1
    config.synthesis.runtime.max_consecutive_category_discards = 1
    config.synthesis.runtime.category_backoff_duration_s = 3600
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _failing_self_consistency_result(
            solution_tool_calls=0,
            facts_match_result=True,
            check_constraints_result=False,
            verify_result=False,
        )

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    with pytest.raises(SynthesisSelfConsistencyError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    status = await runtime.category_status()
    assert status[CategoryTaxonomy.ASSIGNMENT].consecutive_discards == 1
    assert status[CategoryTaxonomy.ASSIGNMENT].backed_off is True
    assert (
        status[CategoryTaxonomy.ASSIGNMENT].last_outcome
        == SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED
    )

    with pytest.raises(SynthesisCategoryBackoffError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    assert exc_info.value.consecutive_discards == 1
    assert exc_info.value.backoff_until > datetime.now(timezone.utc)
    assert exc_info.value.last_outcome == SynthesisSelfConsistencyOutcome.SELF_CONSISTENCY_FAILED
    assert backend.calls.count(SynthesisPhase.ARTIFACT_GENERATION) == 1


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_category_status_resets_after_success(monkeypatch):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 1
    config.synthesis.runtime.max_consecutive_category_discards = 2
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report()

    checks = iter(
        [
            _failing_self_consistency_result(
                solution_tool_calls=0,
                facts_match_result=True,
                check_constraints_result=False,
                verify_result=False,
            ),
            _passing_self_consistency_result(),
        ]
    )

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return next(checks)

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    with pytest.raises(SynthesisSelfConsistencyError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    status = await runtime.category_status()
    assert status[CategoryTaxonomy.ASSIGNMENT].consecutive_discards == 1
    assert status[CategoryTaxonomy.ASSIGNMENT].backed_off is False
    assert status[CategoryTaxonomy.ASSIGNMENT].last_error_codes == []

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert draft.environment.quality_metrics.self_consistency_pass is True
    assert await runtime.category_status() == {}


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_category_status_requires_bound_db_match():
    config = load_config("rl_task_foundry.yaml")
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [] for phase in SynthesisPhase},
    )
    await runtime._bind_db_id("sakila")

    with pytest.raises(SynthesisDbBindingError):
        await runtime.category_status(db_id="northwind")


@pytest.mark.asyncio
async def test_synthesis_runtime_category_status_integrates_with_scheduler(monkeypatch):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 1
    config.synthesis.runtime.max_consecutive_category_discards = 1
    config.synthesis.runtime.category_backoff_duration_s = 600
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _diagnostic_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _failing_self_consistency_result(
            solution_tool_calls=0,
            facts_match_result=True,
            check_constraints_result=False,
            verify_result=False,
        )

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    with pytest.raises(SynthesisSelfConsistencyError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    scheduler = SynthesisDomainScheduler()
    snapshot = SynthesisDbSnapshot(
        db_id="sakila",
        categories=[CategoryTaxonomy.ASSIGNMENT],
        category_status=await runtime.category_status(),
    )
    decision = scheduler.choose_next([snapshot])

    assert decision.status == SynthesisSelectionStatus.BACKOFF
    assert decision.wait_until is not None
    assert decision.wait_seconds > 0


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_wraps_registration_gate_exceptions(monkeypatch):
    config = load_config("rl_task_foundry.yaml").model_copy(deep=True)
    config.synthesis.runtime.max_self_consistency_iterations = 1
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

    with pytest.raises(SynthesisSelfConsistencyError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )
    assert exc_info.value.last_registration_diagnostics is None
    assert exc_info.value.attempts[0].outcome == SynthesisSelfConsistencyOutcome.REGISTRATION_FAILED


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_introspects_graph_once_when_not_provided(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml")
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )
    introspect_calls = 0

    async def _fake_introspect(self):
        nonlocal introspect_calls
        introspect_calls += 1
        return _sample_graph()

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _passing_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    monkeypatch.setattr(
        "rl_task_foundry.synthesis.runtime.PostgresSchemaIntrospector.introspect",
        _fake_introspect,
    )
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)

    draft_one = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
    )
    draft_two = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
    )

    assert draft_one.schema_summary["edge_count"] == 1
    assert draft_two.schema_summary["edge_count"] == 1
    assert introspect_calls == 1


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_is_single_db_per_instance(monkeypatch):
    config = load_config("rl_task_foundry.yaml")
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _passing_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(
        SynthesisAgentRuntime,
        "_run_self_consistency_check",
        _fake_self_consistency,
    )

    await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )
    with pytest.raises(SynthesisDbBindingError):
        await runtime.synthesize_environment_draft(
            db_id="northwind",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_rejects_concurrent_cross_db_binding(monkeypatch):
    config = load_config("rl_task_foundry.yaml")
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={},
    )

    results = await asyncio.gather(
        runtime._bind_db_id("sakila"),
        runtime._bind_db_id("northwind"),
        return_exceptions=True,
    )

    assert sum(result is None for result in results) == 1
    errors = [result for result in results if isinstance(result, BaseException)]
    assert len(errors) == 1
    assert isinstance(errors[0], SynthesisDbBindingError)


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_propagates_materialize_validation_failures(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml")
    backend = _FakeBackend(
        provider_name="codex_oauth",
        model_name="gpt-5.4-mini",
        payloads=_payloads(),
    )
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={phase: [backend] for phase in SynthesisPhase},
    )

    async def _fake_registration_gate(self, *, bundle, proposed_environment):
        return _passing_registration_report()

    async def _fake_self_consistency(self, *, bundle, proposed_environment):
        return _passing_self_consistency_result()

    def _fake_model_validate(payload):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_self_consistency_check", _fake_self_consistency)
    monkeypatch.setattr(
        "rl_task_foundry.synthesis.runtime.EnvironmentContract.model_validate",
        _fake_model_validate,
    )

    with pytest.raises(ValueError, match="forced validation failure"):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_passes_atomic_tool_reference_to_self_consistency_pool():
    config = load_config("rl_task_foundry.yaml")
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={},
    )
    await runtime._bind_db_id("sakila")
    runtime._atomic_tool_bundles["sakila"] = _sample_atomic_tool_bundle("sakila")
    captured: dict[str, object] = {}

    class _FakePool:
        async def run_self_consistency_check(self, **kwargs):
            captured.update(kwargs)
            return _passing_self_consistency_result()

    runtime._registration_pool = _FakePool()

    await runtime._run_self_consistency_check(
        bundle=_sample_artifacts(),
        proposed_environment=_sample_proposed_environment(),
    )

    assert captured["atomic_tool_set_ref"] == "db://sakila"
    assert captured["solution_source"] == _sample_artifacts().solution_source
    assert captured["verifier_source"] == _sample_artifacts().verifier_source
    assert "tool_source" not in captured


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_passes_atomic_tool_reference_to_registration_runner(
    monkeypatch,
):
    config = load_config("rl_task_foundry.yaml")
    runtime = SynthesisAgentRuntime(
        config,
        phase_backends={},
    )
    await runtime._bind_db_id("sakila")
    runtime._atomic_tool_bundles["sakila"] = _sample_atomic_tool_bundle("sakila")
    runtime._registration_pool = SimpleNamespace()
    captured: dict[str, object] = {}

    async def _fake_run_registration_bundle(**kwargs):
        captured.update(kwargs)
        return _passing_registration_report()

    monkeypatch.setattr("rl_task_foundry.synthesis.runtime.run_registration_bundle", _fake_run_registration_bundle)

    await runtime._run_registration_gate(
        bundle=_sample_artifacts(),
        proposed_environment=_sample_proposed_environment(),
    )

    assert captured["atomic_tool_set_ref"] == "db://sakila"
    assert captured["bundle"] == _sample_artifacts()
    assert captured["pool"] is runtime._registration_pool
    assert "tool_source" not in captured


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_uses_structured_output_and_tracing(
    tmp_path,
    monkeypatch,
):
    tracing_disabled: list[bool] = []

    class FakeAsyncOpenAI:
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.calls.append(kwargs)

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.last_instance = self

    class FakeSQLiteSession:
        last_instance = None

        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path
            self.__class__.last_instance = self

    class FakeRunner:
        calls: list[dict[str, object]] = []

        @staticmethod
        async def run(agent, input, max_turns, session=None):
            FakeRunner.calls.append(
                {
                    "agent": agent,
                    "input": input,
                    "max_turns": max_turns,
                    "session": session,
                }
            )
            return SimpleNamespace(
                final_output={
                    "domain_hypothesis": "customer_support",
                    "candidate_categories": ["assignment"],
                    "memory_summary": "schema exploration complete",
                },
                _current_turn=2,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=17,
                        output_tokens=9,
                        total_tokens=26,
                    )
                ),
                new_items=["tool-call(schema_probe)"],
            )

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: tracing_disabled.append(disabled),
        ),
    )

    config = load_config("rl_task_foundry.yaml")
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=config.synthesis.runtime,
        session_db_path=tmp_path / "synthesis_sessions.sqlite",
        traces_dir=tmp_path / "synthesis_traces",
    )

    result = await backend.run_stage(
        SynthesisStageRequest(
            phase=SynthesisPhase.SCHEMA_EXPLORATION,
            db_id="sakila",
            atomic_tool_set_ref="db://sakila",
            available_atomic_tools=[
                {
                    "name": "get_customer_by_id",
                    "description": "Lookup a customer by id.",
                    "params_schema": {"type": "object"},
                    "returns_schema": {"type": "object"},
                }
            ],
            domain_name="customer_support",
            user_role="end user",
            agent_role="organization AI assistant",
            scenario_description="help requests",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            schema_summary={"table_count": 2},
        )
    )

    assert isinstance(result.payload, SchemaExplorationOutput)
    assert result.payload.candidate_categories == [CategoryTaxonomy.ASSIGNMENT]
    assert result.memory_entry.summary == "schema exploration complete"
    assert result.memory_entry.token_usage["total_tokens"] == 26
    assert result.tool_traces[0].tool_name == "schema_probe"
    assert tracing_disabled == [True]
    assert FakeAsyncOpenAI.calls[0]["api_key"] == "dummy"
    assert FakeRunner.calls[0]["max_turns"] == config.synthesis.runtime.max_turns
    assert FakeSQLiteSession.last_instance.session_id == "sakila:schema_exploration:codex_oauth"
    assert FakeAgent.last_instance.kwargs["output_type"] is SchemaExplorationOutput
    assert "available atomic tools" in FakeAgent.last_instance.kwargs["instructions"]
    assert "new tools can be created" in FakeAgent.last_instance.kwargs["instructions"]
    request_input = json.loads(FakeRunner.calls[0]["input"])
    assert request_input["atomic_tool_set_ref"] == "db://sakila"
    assert request_input["available_atomic_tool_names"] == ["get_customer_by_id"]
    assert request_input["available_atomic_tools"][0]["name"] == "get_customer_by_id"
    assert "params_schema" not in request_input["available_atomic_tools"][0]
    assert "returns_schema" not in request_input["available_atomic_tools"][0]
    assert (tmp_path / "synthesis_traces" / "transcripts").exists()


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_relaxes_strict_json_schema_for_artifact_generation(
    tmp_path,
    monkeypatch,
):
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgentOutputSchema:
        last_call = None

        def __init__(self, output_type, strict_json_schema=True):
            self.output_type = output_type
            self.strict_json_schema = strict_json_schema
            self.__class__.last_call = self

    class FakeAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.last_instance = self

    class FakeSQLiteSession:
        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            raise RuntimeError("stop after agent construction")

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AgentOutputSchema=FakeAgentOutputSchema,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: None,
        ),
    )

    config = load_config("rl_task_foundry.yaml")
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=config.synthesis.runtime,
        session_db_path=tmp_path / "synthesis_sessions.sqlite",
        traces_dir=tmp_path / "synthesis_traces",
    )

    with pytest.raises(RuntimeError, match="stop after agent construction"):
        await backend.run_stage(
            SynthesisStageRequest(
                phase=SynthesisPhase.ARTIFACT_GENERATION,
                db_id="sakila",
                atomic_tool_set_ref="db://sakila",
                available_atomic_tools=[],
                domain_name="customer_support",
                user_role="end user",
                agent_role="organization AI assistant",
                scenario_description="help requests",
                requested_category=CategoryTaxonomy.ASSIGNMENT,
                schema_summary={"table_count": 2},
                previous_outputs=_payloads(),
            )
        )

    assert FakeAgentOutputSchema.last_call is not None
    assert FakeAgentOutputSchema.last_call.output_type is ArtifactGenerationOutput
    assert FakeAgentOutputSchema.last_call.strict_json_schema is False
    assert FakeAgent.last_instance.kwargs["output_type"] is FakeAgentOutputSchema.last_call


def test_openai_agents_synthesis_backend_repairs_split_artifact_generation_json() -> None:
    repaired = backend_module._repair_split_json_object(
        '{"proposed_environment":{"task":{}},"verifier":{"entrypoint":"verify"}},'
        '"artifacts":{"solution_source":"x"}}'
    )

    assert repaired == {
        "proposed_environment": {"task": {}},
        "verifier": {"entrypoint": "verify"},
        "artifacts": {"solution_source": "x"},
    }


def test_openai_agents_synthesis_backend_coerces_artifact_generation_fact_shorthand() -> None:
    proposed_environment = _sample_proposed_environment().model_dump(mode="python")
    proposed_environment["verifier"]["facts_schema"] = {
        "facts": [
            {"name": "customer_id", "type": "int"},
            {"name": "total_amount", "type": "number"},
        ]
    }
    proposed_environment["shadow_verifier"]["facts_schema"] = {
        "facts": [
            {"name": "customer_id", "type": "int"},
            {"name": "total_amount", "type": "number"},
        ]
    }
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        atomic_tool_set_ref="db://sakila",
        available_atomic_tools=[],
        domain_name="customer_support",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        previous_outputs=_payloads(),
    )

    payload, repair_codes = backend_module._normalize_phase_payload(
        request,
        {
            "proposed_environment": proposed_environment,
            "artifacts": _sample_artifacts().model_dump(mode="python"),
        },
    )

    assert isinstance(payload, ArtifactGenerationOutput)
    verifier_facts = payload.proposed_environment.verifier.facts_schema.facts
    shadow_facts = payload.proposed_environment.shadow_verifier.facts_schema.facts
    assert verifier_facts[0].key == "customer_id"
    assert verifier_facts[0].entity_ref == "answer"
    assert verifier_facts[0].attribute == "customer_id"
    assert verifier_facts[0].value_type == "int"
    assert shadow_facts[1].key == "total_amount"
    assert shadow_facts[1].value_type == "float"
    assert repair_codes == ["facts_schema_normalized"]


def test_openai_agents_synthesis_backend_reports_split_json_repair_codes() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.CATEGORY_INFERENCE,
        db_id="sakila",
        atomic_tool_set_ref="db://sakila",
        available_atomic_tools=[],
        domain_name="customer_support",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        previous_outputs=_payloads(),
    )

    payload, repair_codes = backend_module._normalize_phase_payload(
        request,
        '{"category":"assignment"},'
        '"validation_notes":"best match",'
        '"memory_summary":"category selected"}',
    )

    assert isinstance(payload, CategoryInferenceOutput)
    assert payload.selected_category == CategoryTaxonomy.ASSIGNMENT
    assert repair_codes == ["split_json_repaired", "category_inference_remapped"]


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_accepts_markdown_fenced_json(
    tmp_path,
    monkeypatch,
):
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSQLiteSession:
        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            return SimpleNamespace(
                final_output="""```json
{"selected_category":"assignment","rationale":"best match","memory_summary":"category selected"}
```""",
                _current_turn=1,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=10,
                        output_tokens=6,
                        total_tokens=16,
                    )
                ),
                new_items=[],
            )

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: None,
        ),
    )

    config = load_config("rl_task_foundry.yaml")
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=config.synthesis.runtime,
        session_db_path=tmp_path / "synthesis_sessions.sqlite",
        traces_dir=tmp_path / "synthesis_traces",
    )

    result = await backend.run_stage(
        SynthesisStageRequest(
            phase=SynthesisPhase.CATEGORY_INFERENCE,
            db_id="sakila",
            domain_name="customer_support",
            user_role="end user",
            agent_role="organization AI assistant",
            scenario_description="help requests",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            schema_summary={"table_count": 2},
            previous_outputs={
                SynthesisPhase.SCHEMA_EXPLORATION: SchemaExplorationOutput(
                    domain_hypothesis="customer_support",
                    candidate_categories=[CategoryTaxonomy.ASSIGNMENT],
                    memory_summary="schema explored",
                )
            },
        )
    )

    assert isinstance(result.payload, CategoryInferenceOutput)
    assert result.payload.selected_category == CategoryTaxonomy.ASSIGNMENT
    assert result.payload_repair_codes == []


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_coerces_schema_exploration_alias_payload(
    tmp_path,
    monkeypatch,
):
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSQLiteSession:
        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            return SimpleNamespace(
                final_output={
                    "task_category": "assignment",
                    "compositionality": "single-hop lookup and traversal tasks",
                    "supported_query_patterns": ["retrieve one row by primary key"],
                    "limitations": ["no arbitrary aggregation"],
                    "domain_fit": "movie rental support workflows",
                },
                _current_turn=1,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=10,
                        output_tokens=6,
                        total_tokens=16,
                    )
                ),
                new_items=[],
            )

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: None,
        ),
    )

    config = load_config("rl_task_foundry.yaml")
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=config.synthesis.runtime,
        session_db_path=tmp_path / "synthesis_sessions.sqlite",
        traces_dir=tmp_path / "synthesis_traces",
    )

    result = await backend.run_stage(
        SynthesisStageRequest(
            phase=SynthesisPhase.SCHEMA_EXPLORATION,
            db_id="sakila",
            atomic_tool_set_ref="db://sakila",
            available_atomic_tools=[],
            domain_name="customer_support",
            user_role="end user",
            agent_role="organization AI assistant",
            scenario_description="help requests",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            schema_summary={"table_count": 2},
        )
    )

    assert isinstance(result.payload, SchemaExplorationOutput)
    assert result.payload.domain_hypothesis == "movie rental support workflows"
    assert result.payload.candidate_categories == [CategoryTaxonomy.ASSIGNMENT]
    assert result.payload_repair_codes == ["schema_exploration_remapped"]


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_rejects_schema_exploration_category_clusters_without_taxonomy_match(
    tmp_path,
    monkeypatch,
):
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSQLiteSession:
        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            return SimpleNamespace(
                final_output={
                    "categories": [
                        {
                            "name": "single_record_lookup",
                            "description": "retrieve one row by primary key",
                        },
                        {
                            "name": "direct_relation_traversal",
                            "description": "follow one-hop foreign keys",
                        },
                    ],
                    "unsupported_categories": [
                        {
                            "name": "aggregation_general",
                            "reason": "no general aggregate tools",
                        }
                    ],
                    "domain_fit": "movie rental support workflows",
                },
                _current_turn=1,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=10,
                        output_tokens=6,
                        total_tokens=16,
                    )
                ),
                new_items=[],
            )

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: None,
        ),
    )

    config = load_config("rl_task_foundry.yaml")
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=config.synthesis.runtime,
        session_db_path=tmp_path / "synthesis_sessions.sqlite",
        traces_dir=tmp_path / "synthesis_traces",
    )

    with pytest.raises(ValidationError):
        await backend.run_stage(
            SynthesisStageRequest(
                phase=SynthesisPhase.SCHEMA_EXPLORATION,
                db_id="sakila",
                atomic_tool_set_ref="db://sakila",
                available_atomic_tools=[],
                domain_name="customer_support",
                user_role="end user",
                agent_role="organization AI assistant",
                scenario_description="help requests",
                requested_category=CategoryTaxonomy.ASSIGNMENT,
                schema_summary={"table_count": 2},
            )
        )


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_coerces_category_inference_alias_payload(
    tmp_path,
    monkeypatch,
):
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSQLiteSession:
        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            return SimpleNamespace(
                final_output={
                    "category": "assignment",
                    "feasible": True,
                    "recommended_task_type": "single-entity lookup with one-hop traversal",
                    "unique_answer_strategy": "pick a unique answer task",
                    "validation_notes": "assignment is supported by the available tool graph",
                    "suggested_constraints": ["keep one-hop traversal"],
                    "attempt_index": 1,
                },
                _current_turn=1,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=10,
                        output_tokens=6,
                        total_tokens=16,
                    )
                ),
                new_items=[],
            )

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: None,
        ),
    )

    config = load_config("rl_task_foundry.yaml")
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=config.synthesis.runtime,
        session_db_path=tmp_path / "synthesis_sessions.sqlite",
        traces_dir=tmp_path / "synthesis_traces",
    )

    result = await backend.run_stage(
        SynthesisStageRequest(
            phase=SynthesisPhase.CATEGORY_INFERENCE,
            db_id="sakila",
            atomic_tool_set_ref="db://sakila",
            available_atomic_tools=[],
            domain_name="customer_support",
            user_role="end user",
            agent_role="organization AI assistant",
            scenario_description="help requests",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            schema_summary={"table_count": 2},
            previous_outputs={
                SynthesisPhase.SCHEMA_EXPLORATION: SchemaExplorationOutput(
                    domain_hypothesis="movie rental support workflows",
                    candidate_categories=[CategoryTaxonomy.ASSIGNMENT],
                    memory_summary="schema exploration completed",
                )
            },
        )
    )

    assert isinstance(result.payload, CategoryInferenceOutput)
    assert result.payload.selected_category == CategoryTaxonomy.ASSIGNMENT
    assert result.payload.rationale == "assignment is supported by the available tool graph"
    assert result.payload_repair_codes == ["category_inference_remapped"]


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_retries_without_structured_output_on_model_behavior_error(
    tmp_path,
    monkeypatch,
):
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        instances: list[object] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.instances.append(self)

    class FakeSQLiteSession:
        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path

    class ModelBehaviorError(RuntimeError):
        pass

    class FakeRunner:
        calls: list[dict[str, object]] = []

        @staticmethod
        async def run(agent, input, max_turns, session=None):
            FakeRunner.calls.append(
                {
                    "agent": agent,
                    "input": input,
                    "max_turns": max_turns,
                    "session": session,
                }
            )
            if agent.kwargs["output_type"] is not None:
                raise ModelBehaviorError("structured output rejected")
            return SimpleNamespace(
                final_output=(
                    '{"domain_hypothesis":"customer_support",'
                    '"candidate_categories":["assignment"],'
                    '"memory_summary":"schema exploration complete"}'
                ),
                _current_turn=1,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=10,
                        output_tokens=6,
                        total_tokens=16,
                    )
                ),
                new_items=[],
            )

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: None,
        ),
    )

    config = load_config("rl_task_foundry.yaml")
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=config.synthesis.runtime,
        session_db_path=tmp_path / "synthesis_sessions.sqlite",
        traces_dir=tmp_path / "synthesis_traces",
    )

    result = await backend.run_stage(
        SynthesisStageRequest(
            phase=SynthesisPhase.SCHEMA_EXPLORATION,
            db_id="sakila",
            atomic_tool_set_ref="db://sakila",
            available_atomic_tools=[
                {
                    "name": "get_customer_by_id",
                    "description": "Lookup a customer by id.",
                    "params_schema": {"type": "object"},
                    "returns_schema": {"type": "object"},
                }
            ],
            domain_name="customer_support",
            user_role="end user",
            agent_role="organization AI assistant",
            scenario_description="help requests",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            schema_summary={"table_count": 2},
        )
    )

    assert isinstance(result.payload, SchemaExplorationOutput)
    assert result.payload.candidate_categories == [CategoryTaxonomy.ASSIGNMENT]
    assert len(FakeRunner.calls) == 2
    assert FakeRunner.calls[0]["agent"].kwargs["output_type"] is SchemaExplorationOutput
    assert FakeRunner.calls[1]["agent"].kwargs["output_type"] is None
    assert FakeRunner.calls[1]["session"] is None


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_writes_normalize_failure_artifact(
    tmp_path,
    monkeypatch,
):
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSQLiteSession:
        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            return SimpleNamespace(
                final_output={"memory_summary": "missing required fields"},
                _current_turn=1,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=10,
                        output_tokens=6,
                        total_tokens=16,
                    )
                ),
                new_items=[],
            )

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: None,
        ),
    )

    config = load_config("rl_task_foundry.yaml")
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=config.synthesis.runtime,
        session_db_path=tmp_path / "synthesis_sessions.sqlite",
        traces_dir=tmp_path / "synthesis_traces",
    )

    with pytest.raises(ValidationError):
        await backend.run_stage(
            SynthesisStageRequest(
                phase=SynthesisPhase.ARTIFACT_GENERATION,
                db_id="sakila",
                atomic_tool_set_ref="db://sakila",
                available_atomic_tools=[],
                domain_name="customer_support",
                user_role="end user",
                agent_role="organization AI assistant",
                scenario_description="help requests",
                requested_category=CategoryTaxonomy.ASSIGNMENT,
                schema_summary={"table_count": 2},
            )
        )

    failure_path = (
        tmp_path
        / "synthesis_traces"
        / "normalize_failures"
        / "sakila__artifact_generation__codex_oauth__gpt-5.4-mini.json"
    )
    assert failure_path.exists()
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["error_type"] == "ValidationError"
    assert payload["phase"] == "artifact_generation"
    assert payload["raw_final_output"] == {"memory_summary": "missing required fields"}
