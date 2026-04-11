from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import DatabaseConfig
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.path_catalog import build_path_catalog
from rl_task_foundry.tasks.models import TaskSpec
from rl_task_foundry.truth.generator import TierAGroundTruthGenerator, _prepare_asyncpg_query
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema


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
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="amount",
                        data_type="numeric",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="shipped_at",
                        data_type="timestamp without time zone",
                        ordinal_position=4,
                        is_nullable=True,
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
    catalog = build_path_catalog(graph, max_hops=2)
    return graph, catalog, catalog.get("orders.shipments")


def _database_config() -> DatabaseConfig:
    return DatabaseConfig(
        dsn="postgresql://example/test",
        schema_allowlist=["public"],
        readonly_role="rlvr_reader",
        statement_timeout_ms=5000,
        lock_timeout_ms=1000,
        idle_tx_timeout_ms=5000,
        solver_pool_size=4,
        control_pool_size=2,
    )


def _task(
    *,
    answer_schema: AnswerSchema,
    outcome_type: str = "answer",
    question_family: str = "status_lookup",
) -> TaskSpec:
    return TaskSpec(
        task_id="task_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question_family=question_family,
        question="현재 배송 상태는 무엇인가요?",
        outcome_type=outcome_type,
        answer_schema=answer_schema,
        selected_path_id="orders.shipments",
        required_hops=1,
        tool_level=1,
        tool_bundle_id="orders.shipments.L1",
        sensitivity_policy="default",
    )


def test_prepare_asyncpg_query_preserves_casts_and_converts_named_params() -> None:
    sql = """
        SELECT t1."shipped_at"::date AS "ship_date"
        FROM "public"."orders" AS t0
        JOIN "public"."shipments" AS t1 ON t0."shipment_id" = t1."shipment_id"
        WHERE t0."order_id" = :anchor_order_id
    """
    prepared_sql, args = _prepare_asyncpg_query(sql, {"anchor_order_id": 1})

    assert "::date" in prepared_sql
    assert ":anchor_order_id" not in prepared_sql
    assert "$1" in prepared_sql
    assert args == [1]


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_builds_scalar_lookup(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    async def _fake_fetch_rows(_sql, _params):
        return [{"delivery_status": "IN_TRANSIT"}]

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    truth = await generator.generate(
        _task(
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="delivery_status",
                        type="string",
                        canonicalizer="lower_trim",
                        source_columns=["shipments.status"],
                    )
                ]
            )
        )
    )

    assert 'DISTINCT ON (t1."status")' in truth.verification_sql
    assert 't1."shipment_id" ASC' in truth.verification_sql
    assert truth.canonical_answer == {"delivery_status": "in_transit"}
    assert truth.row_context == [{"delivery_status": "IN_TRANSIT"}]


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_supports_list_answers(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    async def _fake_fetch_rows(_sql, _params):
        return [
            {"status_history": "shipped"},
            {"status_history": "in_transit"},
        ]

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    truth = await generator.generate(
        _task(
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="status_history",
                        type="list[string]",
                        ordered=True,
                        canonicalizer="lower_trim",
                        source_columns=["shipments.status"],
                    )
                ]
            )
        )
    )

    assert 'DISTINCT ON (t1."status")' in truth.verification_sql
    assert 't1."shipment_id" ASC' in truth.verification_sql
    assert truth.canonical_answer == {"status_history": ["shipped", "in_transit"]}


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_resolves_latest_timeline_value(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    async def _fake_fetch_rows(_sql, _params):
        return [{"latest_shipped_at": datetime(2025, 10, 3, 14, 15, 0)}]

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    truth = await generator.generate(
        _task(
            question_family="timeline_resolution",
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="latest_shipped_at",
                        type="datetime",
                        canonicalizer="datetime",
                        source_columns=["shipments.shipped_at"],
                    )
                ]
            ),
        )
    )

    assert 'ORDER BY t1."shipped_at" DESC' in truth.verification_sql
    assert "LIMIT 1" in truth.verification_sql
    assert truth.canonical_answer == {"latest_shipped_at": "2025-10-03T14:15:00"}


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_supports_count_and_exists(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    count_rows = [{"count": 1}]
    exists_rows = [{"exists": True}]

    async def _fake_fetch_rows(sql, _params):
        if "COUNT(DISTINCT" in sql and "AS count" in sql:
            return count_rows
        return exists_rows

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    count_truth = await generator.generate(
        _task(
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="delivery_count",
                        type="int",
                        canonicalizer="identity",
                        source_columns=["meta:count"],
                    )
                ]
            )
        )
    )
    exists_truth = await generator.generate(
        _task(
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="has_delivery",
                        type="bool",
                        canonicalizer="identity",
                        source_columns=["meta:exists"],
                    )
                ]
            )
        )
    )

    assert count_truth.canonical_answer == {"delivery_count": 1}
    assert exists_truth.canonical_answer == {"has_delivery": True}


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_supports_reverse_count_contract(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    async def _fake_fetch_rows(sql, _params):
        assert 'JOIN "public"."orders" AS r2' in sql
        return [{"count": 3}]

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    truth = await generator.generate(
        _task(
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="related_count",
                        type="int",
                        canonicalizer="identity",
                        source_columns=["meta:count"],
                    )
                ]
            ),
        ).model_copy(
            update={
                "contract_metadata": {
                    "count_mode": "reverse_relation",
                    "count_relation_constraint": "orders_shipments_fk",
                    "count_relation_source_schema": "public",
                    "count_relation_source_table": "orders",
                    "count_reference_schema": "public",
                    "count_reference_table": "shipments",
                }
            }
        )
    )

    assert truth.canonical_answer == {"related_count": 3}


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_supports_no_result_branch(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    async def _fake_fetch_rows(_sql, _params):
        return []

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    truth = await generator.generate(
        _task(
            outcome_type="no_result",
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="delivery_status",
                        type="string",
                        nullable=True,
                        canonicalizer="lower_trim",
                        source_columns=["shipments.status"],
                    )
                ]
            ),
        )
    )

    assert truth.canonical_answer == {"delivery_status": None}
    assert truth.row_context == []


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_normalizes_date_fields(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    async def _fake_fetch_rows(_sql, _params):
        return [{"ship_date": datetime(2026, 4, 11, 15, 30, 0)}]

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    truth = await generator.generate(
        _task(
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="ship_date",
                        type="date",
                        canonicalizer="iso_date",
                        source_columns=["shipments.shipped_at"],
                    )
                ]
            )
        )
    )

    assert '::date AS "ship_date"' in truth.verification_sql
    assert truth.canonical_answer == {"ship_date": "2026-04-11"}


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_uses_configured_float_precision(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
        float_precision=3,
    )

    async def _fake_fetch_rows(_sql, _params):
        return [{"avg_amount": 12.3456}]

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    truth = await generator.generate(
        _task(
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="avg_amount",
                        type="float",
                        canonicalizer="round_custom",
                        source_columns=["shipments.amount"],
                    )
                ]
            )
        )
    )

    assert 'ROUND((t1."amount")::numeric, 3) AS "avg_amount"' in truth.verification_sql
    assert truth.canonical_answer == {"avg_amount": 12.346}


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_rejects_tier_b_task(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    task = _task(
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="delivery_status",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["shipments.status"],
                )
            ]
        )
    ).model_copy(
        update={"label_tier": "B"}
    )

    with pytest.raises(NotImplementedError):
        await generator.generate(task)


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_rejects_ambiguous_scalar(monkeypatch):
    graph, catalog, _path = _graph_and_path()
    generator = TierAGroundTruthGenerator(
        database=_database_config(),
        graph=graph,
        catalog=catalog,
    )

    async def _fake_fetch_rows(_sql, _params):
        return [
            {"delivery_status": "IN_TRANSIT"},
            {"delivery_status": "DELIVERED"},
        ]

    monkeypatch.setattr(generator, "_fetch_rows", _fake_fetch_rows)

    with pytest.raises(ValueError, match="uniquely determined"):
        await generator.generate(
            _task(
                answer_schema=AnswerSchema(
                    fields=[
                        AnswerField(
                            name="delivery_status",
                            type="string",
                            canonicalizer="lower_trim",
                            source_columns=["shipments.status"],
                        )
                    ]
                )
            )
        )


@pytest.mark.asyncio
async def test_tier_a_ground_truth_generator_hits_sakila_customer_city():
    config = load_config(Path("rl_task_foundry.yaml"))
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.privacy.default_visibility,
        visibility_overrides=config.privacy.visibility_overrides,
    )
    graph = await introspector.introspect()
    catalog = build_path_catalog(graph, max_hops=config.tool_compiler.max_hops)
    generator = TierAGroundTruthGenerator(
        database=config.database,
        graph=graph,
        catalog=catalog,
    )

    task = TaskSpec(
        task_id="sakila_customer_city",
        anchor_table="customer",
        anchor_pk_column="customer_id",
        anchor_pk_value="1",
        domain=config.domain.name,
        language=config.domain.language,
        label_tier="A",
        question_family="status_lookup",
        question="현재 고객의 도시 정보는 무엇인가요?",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="customer_city",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["city.city"],
                )
            ]
        ),
        selected_path_id="customer.address.city",
        required_hops=2,
        tool_level=1,
        tool_bundle_id="customer.address.city.L1",
        sensitivity_policy="default",
    )

    truth = await generator.generate(task)

    assert truth.sql_params == {"anchor_customer_id": 1}
    assert 'DISTINCT ON (t2."city")' in truth.verification_sql
    assert 't2."city_id" ASC' in truth.verification_sql
    assert truth.canonical_answer == {"customer_city": "sasebo"}
    assert truth.row_context == [{"customer_city": "Sasebo"}]
