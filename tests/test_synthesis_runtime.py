from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import ModelRef, ProviderConfig
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.synthesis import backend_openai_agents as backend_module
from rl_task_foundry.synthesis.backend_openai_agents import OpenAIAgentsSynthesisBackend
from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    ConstraintKind,
    ConstraintSummaryItem,
    EnvironmentContract,
    EnvironmentStatus,
    MaterializedFactsSchema,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    ShadowVerifierContract,
    SolutionContract,
    TaskContract,
    ToolContract,
    ToolSelfTestContract,
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
from rl_task_foundry.synthesis.runtime import (
    ArtifactGenerationOutput,
    CURRENT_SYNTHESIS_GENERATOR_VERSION,
    CategoryInferenceOutput,
    ProposedEnvironmentDraft,
    RUNTIME_OWNED_ENVIRONMENT_FIELDS,
    SchemaExplorationOutput,
    SynthesisAgentRuntime,
    SynthesisCategoryMismatchError,
    SynthesisDbBindingError,
    SynthesisMemoryEntry,
    SynthesisPhase,
    SynthesisRegistrationError,
    SynthesisStageRequest,
    SynthesisStageResult,
    SynthesisToolTraceEntry,
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


def _sample_proposed_environment(
    category: CategoryTaxonomy = CategoryTaxonomy.ASSIGNMENT,
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
        tools=[
            ToolContract(
                name="get_customer_assignments",
                description="Return candidate assignments for a customer.",
                return_schema=OutputFieldContract(
                    name="rows",
                    type=OutputFieldType.LIST,
                    items=OutputFieldContract(
                        name="row",
                        type=OutputFieldType.OBJECT,
                        fields=[OutputFieldContract(name="customer", type=OutputFieldType.STRING)],
                    ),
                ),
            )
        ],
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
            difficulty_vector={},
        ),
        solution=SolutionContract(),
        tool_self_test=ToolSelfTestContract(),
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
        tool_source="async def get_customer_assignments(conn, customer_id):\n    return []\n",
        tool_self_test_source="async def run_self_test(tools):\n    return {'ok': True}\n",
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


def _failing_registration_report() -> RegistrationBundleReport:
    report = _passing_registration_report()
    return report.model_copy(update={"status": RegistrationBundleStatus.FAILED})


class _FakeBackend:
    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        payloads: dict[SynthesisPhase, object],
        fail_phases: set[SynthesisPhase] | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
        self.payloads = payloads
        self.fail_phases = fail_phases or set()
        self.calls: list[SynthesisPhase] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def run_stage(self, request: SynthesisStageRequest) -> SynthesisStageResult:
        self.calls.append(request.phase)
        if request.phase in self.fail_phases:
            raise RuntimeError(f"{self.provider_name} failed")
        payload = self.payloads[request.phase]
        return SynthesisStageResult(
            phase=request.phase,
            provider=self.provider_name,
            model=self.model_name,
            payload=payload,
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
        return _passing_registration_report()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert draft.selected_category == CategoryTaxonomy.ASSIGNMENT
    assert draft.environment.status == EnvironmentStatus.DRAFT
    assert draft.environment.db_id == "sakila"
    assert draft.environment.domain == config.domain.name
    assert draft.environment.generator_version == CURRENT_SYNTHESIS_GENERATOR_VERSION
    assert draft.environment.env_id.startswith("env_assignment_")
    assert len(draft.environment.env_id.split("_")[-1]) == 16
    assert draft.environment.quality_metrics.self_consistency_pass is False
    assert draft.environment.tool_signature.startswith("sha256:")
    assert draft.environment.task_signature.startswith("sha256:")
    assert draft.environment.verifier_signature.startswith("sha256:")
    assert draft.registration_report.status == RegistrationBundleStatus.PASSED
    assert [entry.phase for entry in draft.memory] == list(SynthesisPhase)
    assert draft.provider_status["codex_oauth"].observed_at.tzinfo is not None


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

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

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
    config = load_config("rl_task_foundry.yaml")
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

    with pytest.raises(SynthesisCategoryMismatchError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_rejects_failed_registration(monkeypatch):
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
        return _failing_registration_report()

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

    with pytest.raises(SynthesisRegistrationError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_wraps_registration_gate_exceptions(monkeypatch):
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
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

    with pytest.raises(SynthesisRegistrationError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )


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

    monkeypatch.setattr(
        "rl_task_foundry.synthesis.runtime.PostgresSchemaIntrospector.introspect",
        _fake_introspect,
    )
    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

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

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)

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

    def _fake_model_validate(payload):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_registration_gate", _fake_registration_gate)
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
    assert (tmp_path / "synthesis_traces" / "transcripts").exists()


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
