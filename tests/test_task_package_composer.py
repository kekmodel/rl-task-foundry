from __future__ import annotations

from dataclasses import replace

import pytest

from rl_task_foundry.config.models import DomainConfig, ModelRef, ProviderConfig
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.path_catalog import build_path_catalog
from rl_task_foundry.tasks.composer import ComposeRequest, TaskComposer, _question_policy_violations
from rl_task_foundry.tasks.package_validation import TaskPackageJudgeResult
from rl_task_foundry.tasks.models import TaskSpec
from rl_task_foundry.tools.compiler import compile_canonical_tool_bundle, compile_path_tools
from rl_task_foundry.tools.model_naming import ToolNamingGenerationError
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema


async def _passing_rubric_validation(**_kwargs):
    return TaskPackageJudgeResult(
        pass_validation=True,
        criterion_scores={
            "natural_language": 1,
            "no_answer_leak": 1,
            "no_schema_exposure": 1,
            "semantic_coherence": 1,
            "tool_answerability": 1,
        },
        failures=[],
        summary="ok",
    )


def _graph_and_path():
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="shipment_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="shipments",
                primary_key=("shipment_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="shipment_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="status",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_shipments_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("shipment_id",),
                target_schema="public",
                target_table="shipments",
                target_columns=("shipment_id",),
            )
        ],
    )
    path = build_path_catalog(graph, max_hops=2).get("orders.shipments")
    return graph, path


def _task(*, tool_level: int, question: str) -> TaskSpec:
    return TaskSpec(
        task_id="task_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question_family="status_lookup",
        question=question,
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="delivery_status",
                    type="string",
                    canonicalizer="lower_trim",
                )
            ]
        ),
        selected_path_id="orders.shipments",
        required_hops=1,
        tool_level=tool_level,
        tool_bundle_id="orders.shipments.L1",
        sensitivity_policy="default",
    )


@pytest.mark.asyncio
async def test_task_composer_uses_canonical_bundle_for_l1():
    graph, path = _graph_and_path()
    canonical_bundle = compile_canonical_tool_bundle(graph, path, max_list_cardinality=5)
    composer = TaskComposer(domain=DomainConfig(name="customer_support", language="ko"))

    package = await composer.compose(
        ComposeRequest(
            graph=graph,
            task=_task(tool_level=1, question="현재 배송 상태는 무엇인가요?"),
            path=path,
            canonical_bundle=canonical_bundle,
        )
    )

    assert package.task.presented_tool_bundle_id == package.presented_tool_bundle.bundle_id
    assert package.presented_tool_bundle.canonical_bundle_id == canonical_bundle.bundle_id
    assert package.presented_tool_bundle.generation_metadata["presentation_strategy"] == "canonical_rule_based"
    assert {tool.name_source for tool in package.presented_tool_bundle.tools} == {"rule_based"}
    assert package.available_tool_levels == [1]
    assert [bundle.tool_level for bundle in package.presentation_options] == [1]


@pytest.mark.asyncio
async def test_task_composer_generates_task_aware_l2_presentation(monkeypatch):
    graph, path = _graph_and_path()
    canonical_bundle = compile_canonical_tool_bundle(graph, path, max_list_cardinality=5)
    captured: dict[str, object] = {}

    async def _fake_generate_task_question(**kwargs):
        captured["question_context"] = kwargs["question_context"]
        return "제 주문 배송이 지금 어디쯤 왔는지 알려줘."

    async def _fake_generate_named_tool_bundle(**kwargs):
        captured.update(kwargs)
        bundle = kwargs["bundle"]
        return replace(
            bundle,
            tools=[
                replace(
                    tool,
                    name=f"task_fit_{index}_{tool.kind}",
                    description=f"task-aware {tool.kind}",
                    name_source="model_generated",
                )
                for index, tool in enumerate(bundle.tools, start=1)
            ],
        )

    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_task_question",
        _fake_generate_task_question,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_named_tool_bundle",
        _fake_generate_named_tool_bundle,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.judge_task_package",
        _passing_rubric_validation,
    )

    composer = TaskComposer(
        domain=DomainConfig(name="customer_support", language="ko"),
        provider=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
    )

    package = await composer.compose(
        ComposeRequest(
            graph=graph,
            task=_task(tool_level=2, question="배송 상태와 관련된 항목을 확인해줘."),
            path=path,
            canonical_bundle=canonical_bundle,
            question_context={
                "language": "ko",
                "relationship_depth": 1,
                "forbidden_markers": [
                    {"normalized": "sasebo", "match_mode": "substring", "display": "sasebo"}
                ],
            },
            distractor_bundles=[compile_path_tools(graph, path, tool_level=2, max_list_cardinality=5)],
        )
    )

    assert package.task.question == "제 주문 배송이 지금 어디쯤 왔는지 알려줘."
    assert package.task.question_source == "model_generated"
    assert package.task.question_generation_metadata["status"] == "accepted"
    assert package.task.question_generation_metadata["task_package_judge"]["pass_validation"] is True
    assert package.task.provenance_requirements == ["semantic_key:orders.shipments:lookup"]
    assert captured["task_question"] == "제 주문 배송이 지금 어디쯤 왔는지 알려줘."
    assert captured["question_family"] == "status_lookup"
    assert captured["outcome_type"] == "answer"
    assert captured["bundle"].tool_level == 2
    assert "forbidden_markers" not in captured["question_context"]
    assert package.presented_tool_bundle.generation_metadata["presentation_strategy"] == "task_context_model_generated"
    assert package.presented_tool_bundle.generation_metadata["generated_policy_violations"] == []
    assert package.available_tool_levels == [1, 2]
    assert [bundle.tool_level for bundle in package.presentation_options] == [1, 2]
    assert {tool.name_source for tool in package.presented_tool_bundle.tools if tool.presentation_role == "core"} == {
        "model_generated"
    }
    assert any(tool.presentation_role == "distractor" for tool in package.presented_tool_bundle.tools)


def test_question_policy_rejects_raw_english_schema_tokens_in_korean():
    violations = _question_policy_violations(
        "제 inventory에 등록된 film 제목이 무엇인지 알려주세요.",
        {
            "language": "ko",
            "path_entity_labels": ["inventory", "film"],
            "answer_fields": [{"name": "film_title", "label": "film title"}],
        },
    )

    assert any("raw english schema tokens" in violation for violation in violations)


@pytest.mark.asyncio
async def test_task_composer_falls_back_to_alias_bundle_when_model_generation_fails(monkeypatch):
    graph, path = _graph_and_path()
    canonical_bundle = compile_canonical_tool_bundle(graph, path, max_list_cardinality=5)
    fallback_bundle = compile_path_tools(graph, path, tool_level=2, max_list_cardinality=5)

    async def _fake_generate_named_tool_bundle(**_kwargs):
        raise ToolNamingGenerationError("model output invalid")

    async def _fake_generate_task_question(**_kwargs):
        return "담당 배송 상태를 확인하고 싶어요."

    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_task_question",
        _fake_generate_task_question,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_named_tool_bundle",
        _fake_generate_named_tool_bundle,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.judge_task_package",
        _passing_rubric_validation,
    )

    composer = TaskComposer(
        domain=DomainConfig(name="customer_support", language="ko"),
        provider=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
    )

    package = await composer.compose(
        ComposeRequest(
            graph=graph,
            task=_task(tool_level=2, question="담당 인력 정보를 확인해줘."),
            path=path,
            canonical_bundle=canonical_bundle,
            question_context={"path_id": path.path_id},
            fallback_presented_bundle=fallback_bundle,
        )
    )

    assert package.task.question == "담당 배송 상태를 확인하고 싶어요."
    assert package.task.question_source == "model_generated"
    assert package.presented_tool_bundle.generation_metadata["presentation_strategy"] == "task_context_fallback_alias"
    assert "model output invalid" in str(
        package.presented_tool_bundle.generation_metadata["naming_generation_error"]
    )
    assert package.available_tool_levels == [1, 2]
    assert {tool.name_source for tool in package.presented_tool_bundle.tools} == {"fallback_alias"}


@pytest.mark.asyncio
async def test_task_composer_falls_back_when_generated_l2_fails_quality_gate(monkeypatch):
    graph, path = _graph_and_path()
    canonical_bundle = compile_canonical_tool_bundle(graph, path, max_list_cardinality=5)
    fallback_bundle = compile_path_tools(graph, path, tool_level=2, max_list_cardinality=5)

    async def _fake_generate_named_tool_bundle(**kwargs):
        bundle = kwargs["bundle"]
        return replace(
            bundle,
            tools=[
                replace(
                    tool,
                    name=f"inspect_orders_shipments_{index}",
                    description="too literal",
                    name_source="model_generated",
                )
                for index, tool in enumerate(bundle.tools, start=1)
            ],
        )

    async def _fake_generate_task_question(**_kwargs):
        return "배송 상태를 확인해줘."

    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_task_question",
        _fake_generate_task_question,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_named_tool_bundle",
        _fake_generate_named_tool_bundle,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.judge_task_package",
        _passing_rubric_validation,
    )

    composer = TaskComposer(
        domain=DomainConfig(name="customer_support", language="ko"),
        provider=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
    )

    package = await composer.compose(
        ComposeRequest(
            graph=graph,
            task=_task(tool_level=2, question="배송 상태를 확인해줘."),
            path=path,
            canonical_bundle=canonical_bundle,
            question_context={"path_id": path.path_id},
            fallback_presented_bundle=fallback_bundle,
        )
    )

    assert package.presented_tool_bundle.generation_metadata["presentation_strategy"] == "task_context_fallback_alias"
    assert (
        package.presented_tool_bundle.generation_metadata["naming_generation_error"]
        == "model-generated naming did not pass L2 quality gate"
    )
    assert package.presented_tool_bundle.generation_metadata["generated_policy_violations"]
    assert {tool.name_source for tool in package.presented_tool_bundle.tools} == {"fallback_alias"}


@pytest.mark.asyncio
async def test_task_composer_rejects_question_with_forbidden_term(monkeypatch):
    graph, path = _graph_and_path()
    canonical_bundle = compile_canonical_tool_bundle(graph, path, max_list_cardinality=5)

    async def _fake_generate_task_question(**_kwargs):
        return "제 배송이 sasebo 상태인지 확인해줘."

    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_task_question",
        _fake_generate_task_question,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.judge_task_package",
        _passing_rubric_validation,
    )

    composer = TaskComposer(
        domain=DomainConfig(name="customer_support", language="ko"),
        provider=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
    )

    package = await composer.compose(
        ComposeRequest(
            graph=graph,
            task=_task(tool_level=1, question="배송 상태를 확인해줘."),
            path=path,
            canonical_bundle=canonical_bundle,
            question_context={
                "forbidden_markers": [
                    {"normalized": "sasebo", "match_mode": "substring", "display": "sasebo"}
                ]
            },
        )
    )

    assert package.task.question == "배송 상태를 확인해줘."
    assert package.task.question_source == "seed_fallback"
    assert package.task.question_generation_metadata["status"] == "fallback"


@pytest.mark.asyncio
async def test_task_composer_falls_back_when_llm_rubric_rejects_question(monkeypatch):
    graph, path = _graph_and_path()
    canonical_bundle = compile_canonical_tool_bundle(graph, path, max_list_cardinality=5)

    async def _fake_generate_task_question(**_kwargs):
        return "제 주문 배송이 지금 어디쯤 왔는지 알려줘."

    async def _failing_rubric_validation(**_kwargs):
        return TaskPackageJudgeResult(
            pass_validation=False,
            criterion_scores={
                "natural_language": 1,
                "no_answer_leak": 1,
                "no_schema_exposure": 1,
                "semantic_coherence": 0,
                "tool_answerability": 1,
            },
            failures=["semantic_coherence: asks for location but answer schema is status"],
            summary="semantic mismatch",
        )

    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_task_question",
        _fake_generate_task_question,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.judge_task_package",
        _failing_rubric_validation,
    )

    composer = TaskComposer(
        domain=DomainConfig(name="customer_support", language="ko"),
        provider=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
    )

    seed_question = "배송 상태를 확인해줘."
    package = await composer.compose(
        ComposeRequest(
            graph=graph,
            task=_task(tool_level=1, question=seed_question),
            path=path,
            canonical_bundle=canonical_bundle,
            question_context={"relationship_depth": 1},
        )
    )

    assert package.task.question == seed_question
    assert package.task.question_source == "seed_fallback"
    assert package.task.question_generation_metadata["fallback_reason"] == "task_package_judge_rejected"
    assert package.task.question_generation_metadata["task_package_judge"]["pass_validation"] is False


@pytest.mark.asyncio
async def test_task_composer_retries_question_after_task_package_judge_rejection(monkeypatch):
    graph, path = _graph_and_path()
    canonical_bundle = compile_canonical_tool_bundle(graph, path, max_list_cardinality=5)
    generated_questions = iter(
        [
            "제 주문 배송이 지금 어디쯤 왔는지 알려줘.",
            "지금 제 주문 배송 상태가 어떻게 되어 있는지 알려주세요.",
        ]
    )
    judge_calls = {"count": 0}

    async def _fake_generate_task_question(**_kwargs):
        return next(generated_questions)

    async def _judge_with_retry(**_kwargs):
        judge_calls["count"] += 1
        if judge_calls["count"] == 1:
            return TaskPackageJudgeResult(
                pass_validation=False,
                criterion_scores={
                    "natural_language": 1,
                    "no_answer_leak": 1,
                    "no_schema_exposure": 1,
                    "semantic_coherence": 0,
                    "tool_answerability": 1,
                },
                failures=["semantic_coherence: asks for location but answer schema is status"],
                summary="semantic mismatch",
            )
        return TaskPackageJudgeResult(
            pass_validation=True,
            criterion_scores={
                "natural_language": 1,
                "no_answer_leak": 1,
                "no_schema_exposure": 1,
                "semantic_coherence": 1,
                "tool_answerability": 1,
            },
            failures=[],
            summary="ok",
        )

    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.generate_task_question",
        _fake_generate_task_question,
    )
    monkeypatch.setattr(
        "rl_task_foundry.tasks.composer.judge_task_package",
        _judge_with_retry,
    )

    composer = TaskComposer(
        domain=DomainConfig(name="customer_support", language="ko"),
        provider=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
    )

    package = await composer.compose(
        ComposeRequest(
            graph=graph,
            task=_task(tool_level=1, question="배송 상태를 확인해줘."),
            path=path,
            canonical_bundle=canonical_bundle,
            question_context={"relationship_depth": 1},
        )
    )

    assert package.task.question == "지금 제 주문 배송 상태가 어떻게 되어 있는지 알려주세요."
    assert package.task.question_source == "model_generated"
    assert package.task.question_generation_metadata["task_package_judge"]["pass_validation"] is True
    assert package.task.question_generation_metadata["task_package_judge_attempts"] == 2
    assert len(package.task.question_generation_metadata["task_package_judge_runs"]) == 2
