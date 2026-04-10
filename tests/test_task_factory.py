from __future__ import annotations

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import (
    DomainConfig,
    TaskComposerConfig,
    ToolCompilerConfig,
    VerificationConfig,
)
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.path_catalog import build_path_catalog
from rl_task_foundry.tasks.factory import TaskContractDraft, TierATaskFactory
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema


def _synthetic_graph() -> tuple[SchemaGraph, object]:
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
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="shipped_at",
                        data_type="timestamp without time zone",
                        ordinal_position=3,
                        is_nullable=True,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="city_id",
                        data_type="int4",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="cities",
                primary_key=("city_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="cities",
                        column_name="city_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="cities",
                        column_name="city",
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
            ),
            ForeignKeyEdge(
                constraint_name="shipments_cities_fk",
                source_schema="public",
                source_table="shipments",
                source_columns=("city_id",),
                target_schema="public",
                target_table="cities",
                target_columns=("city_id",),
            ),
        ],
    )
    catalog = build_path_catalog(graph, max_hops=3)
    return graph, catalog


@pytest.mark.asyncio
async def test_task_factory_generates_task_specs_from_synthetic_graph(monkeypatch):
    graph, catalog = _synthetic_graph()
    factory = TierATaskFactory(
        database=load_config("rl_task_foundry.yaml").database,
        domain=DomainConfig(name="customer_support", language="ko"),
        task_config=TaskComposerConfig(
            label_tier="A",
            question_families=["status_lookup", "causal_chain", "timeline_resolution", "aggregate_verification"],
            selected_tool_level=1,
            negative_outcome_ratio=0.0,
            max_attempts_per_anchor=6,
        ),
        tool_compiler=ToolCompilerConfig(
            max_hops=3,
            allow_aggregates=True,
            allow_timelines=True,
            max_list_cardinality=20,
        ),
        verification=VerificationConfig(float_precision=6),
    )

    async def _fake_sample_anchor_values(graph, path, contract, *, limit):
        if contract.outcome_type == "no_result":
            return []
        return ["101"]

    async def _fake_validate_ground_truth(generator, task):
        return True

    async def _patched_sample_anchor_values(self, graph, path, contract, *, limit):
        return await _fake_sample_anchor_values(graph, path, contract, limit=limit)

    async def _patched_validate_ground_truth(self, generator, task):
        return await _fake_validate_ground_truth(generator, task)

    monkeypatch.setattr(TierATaskFactory, "_sample_anchor_values", _patched_sample_anchor_values)
    monkeypatch.setattr(TierATaskFactory, "_validate_ground_truth", _patched_validate_ground_truth)

    tasks = await factory.generate(
        graph,
        catalog,
        limit=6,
        path_ids=["orders.shipments", "orders.shipments.cities"],
    )

    assert tasks
    assert {task.question_family for task in tasks} >= {
        "status_lookup",
        "causal_chain",
        "timeline_resolution",
        "aggregate_verification",
    }
    assert all(task.label_tier == "A" for task in tasks)
    assert all(task.tool_level == 1 for task in tasks)
    assert all(task.outcome_type == "answer" for task in tasks)


def test_task_factory_renders_question_from_contract():
    graph, catalog = _synthetic_graph()
    path = catalog.get("orders.shipments.cities")
    factory = TierATaskFactory(
        database=load_config("rl_task_foundry.yaml").database,
        domain=DomainConfig(name="customer_support", language="en"),
        task_config=TaskComposerConfig(
            label_tier="A",
            question_families=["status_lookup"],
            selected_tool_level=1,
            negative_outcome_ratio=0.0,
            max_attempts_per_anchor=6,
        ),
        tool_compiler=ToolCompilerConfig(
            max_hops=3,
            allow_aggregates=True,
            allow_timelines=True,
            max_list_cardinality=20,
        ),
        verification=VerificationConfig(float_precision=6),
    )
    contract = TaskContractDraft(
        question_family="status_lookup",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="city",
                    type="string",
                    canonicalizer="string_normalized",
                    source_columns=["cities.city"],
                )
            ]
        ),
        field_label="city",
        target_label="cities",
    )

    task = factory._build_task_spec(
        graph=graph,
        path=path,
        contract=contract,
        anchor_pk_value="101",
    )

    assert not hasattr(contract, "question")
    assert task.question == "Check the related city value."


def test_task_factory_builds_list_shaped_causal_chain_contract():
    graph, catalog = _synthetic_graph()
    path = catalog.get("orders.shipments.cities")
    factory = TierATaskFactory(
        database=load_config("rl_task_foundry.yaml").database,
        domain=DomainConfig(name="customer_support", language="en"),
        task_config=TaskComposerConfig(
            label_tier="A",
            question_families=["causal_chain"],
            selected_tool_level=1,
            negative_outcome_ratio=0.0,
            max_attempts_per_anchor=6,
        ),
        tool_compiler=ToolCompilerConfig(
            max_hops=3,
            allow_aggregates=True,
            allow_timelines=True,
            max_list_cardinality=20,
        ),
        verification=VerificationConfig(float_precision=6),
    )

    contracts = factory._contract_drafts_for_path(graph, path)
    list_contracts = [
        contract
        for contract in contracts
        if contract.question_family == "causal_chain"
        and contract.answer_schema.fields[0].type == "list[string]"
    ]

    assert list_contracts
    assert list_contracts[0].answer_schema.fields[0].name == "city_list"
    assert "which related city items exist" in factory._render_en_question(list_contracts[0]).lower()


def test_task_factory_prefers_aggregate_paths_with_count_gt_one():
    graph, catalog = _synthetic_graph()
    path = catalog.get("orders.shipments.cities")
    factory = TierATaskFactory(
        database=load_config("rl_task_foundry.yaml").database,
        domain=DomainConfig(name="customer_support", language="en"),
        task_config=TaskComposerConfig(
            label_tier="A",
            question_families=["aggregate_verification"],
            selected_tool_level=1,
            negative_outcome_ratio=0.0,
            max_attempts_per_anchor=6,
        ),
        tool_compiler=ToolCompilerConfig(
            max_hops=3,
            allow_aggregates=True,
            allow_timelines=True,
            max_list_cardinality=20,
        ),
        verification=VerificationConfig(float_precision=6),
    )
    contract = TaskContractDraft(
        question_family="aggregate_verification",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="related_count",
                    type="int",
                    canonicalizer="int_cast",
                    source_columns=["meta:count"],
                )
            ]
        ),
        target_label="city",
    )

    sql = factory._compile_anchor_sampling_sql(graph, path, contract, limit=5)

    assert "GROUP BY" in sql
    assert "HAVING COUNT(DISTINCT" in sql


def test_task_factory_prefers_list_paths_with_multiple_distinct_values():
    graph, catalog = _synthetic_graph()
    path = catalog.get("orders.shipments.cities")
    factory = TierATaskFactory(
        database=load_config("rl_task_foundry.yaml").database,
        domain=DomainConfig(name="customer_support", language="en"),
        task_config=TaskComposerConfig(
            label_tier="A",
            question_families=["causal_chain"],
            selected_tool_level=1,
            negative_outcome_ratio=0.0,
            max_attempts_per_anchor=6,
        ),
        tool_compiler=ToolCompilerConfig(
            max_hops=3,
            allow_aggregates=True,
            allow_timelines=True,
            max_list_cardinality=20,
        ),
        verification=VerificationConfig(float_precision=6),
    )
    contract = [
        contract
        for contract in factory._contract_drafts_for_path(graph, path)
        if contract.question_family == "causal_chain"
        and contract.answer_schema.fields[0].type == "list[string]"
    ][0]

    sql = factory._compile_anchor_sampling_sql(graph, path, contract, limit=5)

    assert "GROUP BY" in sql
    assert 'COUNT(DISTINCT t2."city") > 1' in sql


def test_task_factory_ignores_non_user_visible_answer_columns():
    factory = TierATaskFactory(
        database=load_config("rl_task_foundry.yaml").database,
        domain=DomainConfig(name="customer_support", language="en"),
        task_config=TaskComposerConfig(
            label_tier="A",
            question_families=["status_lookup"],
            selected_tool_level=1,
            negative_outcome_ratio=0.0,
            max_attempts_per_anchor=6,
        ),
        tool_compiler=ToolCompilerConfig(
            max_hops=3,
            allow_aggregates=True,
            allow_timelines=True,
            max_list_cardinality=20,
        ),
        verification=VerificationConfig(float_precision=6),
    )
    blocked_column = ColumnProfile(
        schema_name="public",
        table_name="languages",
        column_name="name",
        data_type="text",
        ordinal_position=2,
        is_nullable=False,
        visibility="blocked",
    )

    assert factory._is_candidate_answer_column(blocked_column) is False


def test_task_factory_promotes_safe_internal_field_and_contextualizes_generic_name():
    graph, catalog = _synthetic_graph()
    path = catalog.get("orders.shipments.cities")
    factory = TierATaskFactory(
        database=load_config("rl_task_foundry.yaml").database,
        domain=DomainConfig(name="customer_support", language="en"),
        task_config=TaskComposerConfig(
            label_tier="A",
            question_families=["status_lookup"],
            selected_tool_level=1,
            negative_outcome_ratio=0.0,
            max_attempts_per_anchor=6,
        ),
        tool_compiler=ToolCompilerConfig(
            max_hops=3,
            allow_aggregates=True,
            allow_timelines=True,
            max_list_cardinality=20,
        ),
        verification=VerificationConfig(float_precision=6),
    )
    internal_column = ColumnProfile(
        schema_name="public",
        table_name="languages",
        column_name="name",
        data_type="text",
        ordinal_position=2,
        is_nullable=False,
        visibility="internal",
    )

    assert factory._answer_field_visibility(internal_column) == "user_visible"
    assert factory._is_candidate_answer_column(internal_column) is True
    assert factory._answer_field_name(path, internal_column) == "city_name"


@pytest.mark.asyncio
async def test_task_factory_generates_valid_sakila_tasks():
    config = load_config("rl_task_foundry.yaml")
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.privacy.default_visibility,
        visibility_overrides=config.privacy.visibility_overrides,
    )
    graph = await introspector.introspect()
    catalog = build_path_catalog(graph, max_hops=config.tool_compiler.max_hops)
    factory = TierATaskFactory(
        database=config.database,
        domain=config.domain,
        task_config=config.task_composer.model_copy(
            update={
                "question_families": ["status_lookup", "aggregate_verification", "causal_chain"],
                "negative_outcome_ratio": 0.0,
            }
        ),
        tool_compiler=config.tool_compiler,
        verification=config.verification,
    )

    tasks = await factory.generate(
        graph,
        catalog,
        limit=3,
        path_ids=[
            "customer.address.city",
            "rental.inventory.film.language",
        ],
    )

    assert tasks
    assert len(tasks) <= 3
    assert all(task.tool_bundle_id for task in tasks)
    assert all(task.question.strip() for task in tasks)
    assert all(task.outcome_type == "answer" for task in tasks)
    assert {"status_lookup", "causal_chain", "aggregate_verification"} & {
        task.question_family for task in tasks
    }
