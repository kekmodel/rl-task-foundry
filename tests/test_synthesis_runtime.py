from __future__ import annotations

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
from rl_task_foundry.synthesis.registration_runner import GeneratedArtifactBundle
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisCategoryMismatchError,
    SynthesisMemoryEntry,
    SynthesisPhase,
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


def _sample_environment(category: CategoryTaxonomy = CategoryTaxonomy.ASSIGNMENT) -> EnvironmentContract:
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
    return EnvironmentContract(
        env_id="env_assignment_001",
        db_id="sakila",
        domain="customer_support",
        category=category,
        difficulty_vector={},
        created_at=datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc),
        generator_version="milestone-3",
        tool_signature="tool_sig_001",
        task_signature="task_sig_001",
        verifier_signature="verifier_sig_001",
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


class _FakeBackend:
    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        payloads: dict[SynthesisPhase, dict[str, object]],
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
                summary=str(payload.get("memory_summary", f"{request.phase.value} done")),
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


def _payloads(*, category: CategoryTaxonomy = CategoryTaxonomy.ASSIGNMENT) -> dict[SynthesisPhase, dict[str, object]]:
    return {
        SynthesisPhase.SCHEMA_EXPLORATION: {
            "domain_hypothesis": "customer_support",
            "candidate_categories": [category.value],
            "memory_summary": "schema explored",
        },
        SynthesisPhase.CATEGORY_INFERENCE: {
            "selected_category": category.value,
            "rationale": "best match",
            "memory_summary": "category selected",
        },
        SynthesisPhase.ARTIFACT_GENERATION: {
            "environment": _sample_environment(category).model_dump(mode="json"),
            "artifacts": _sample_artifacts().model_dump(mode="json"),
            "memory_summary": "artifacts generated",
        },
    }


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_builds_environment_draft():
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

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert draft.environment.env_id == "env_assignment_001"
    assert draft.selected_category == CategoryTaxonomy.ASSIGNMENT
    assert draft.schema_summary["table_count"] == 2
    assert [entry.phase for entry in draft.memory] == list(SynthesisPhase)
    assert len(draft.tool_traces) == 3


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_reuses_provider_resilience_for_fallback():
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

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert draft.environment.env_id == "env_assignment_001"
    assert failing.calls == [
        SynthesisPhase.SCHEMA_EXPLORATION,
        SynthesisPhase.CATEGORY_INFERENCE,
    ]
    assert fallback.calls == list(SynthesisPhase)
    assert runtime.provider_status()["codex_oauth"].available is False
    assert runtime.provider_status()["codex_oauth"].failures == 2


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_rejects_category_mismatch():
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

    with pytest.raises(SynthesisCategoryMismatchError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_introspects_graph_when_not_provided(monkeypatch):
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

    async def _fake_introspect(self):
        return _sample_graph()

    monkeypatch.setattr(
        "rl_task_foundry.synthesis.runtime.PostgresSchemaIntrospector.introspect",
        _fake_introspect,
    )

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
    )

    assert draft.schema_summary["edge_count"] == 1


@pytest.mark.asyncio
async def test_openai_agents_synthesis_backend_returns_stage_result(tmp_path, monkeypatch):
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
                    "memory_summary": "schema exploration complete",
                    "candidate_categories": ["assignment"],
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

    assert result.phase == SynthesisPhase.SCHEMA_EXPLORATION
    assert result.payload["candidate_categories"] == ["assignment"]
    assert result.memory_entry.summary == "schema exploration complete"
    assert result.memory_entry.token_usage["total_tokens"] == 26
    assert result.tool_traces[0].tool_name == "schema_probe"
    assert FakeAsyncOpenAI.calls[0]["api_key"] == "dummy"
    assert FakeRunner.calls[0]["max_turns"] == config.synthesis.runtime.max_turns
    assert FakeSQLiteSession.last_instance.session_id == "sakila:schema_exploration:codex_oauth"
    assert "schema summary" in FakeAgent.last_instance.kwargs["instructions"].lower()
    assert (tmp_path / "synthesis_traces" / "transcripts").exists()
