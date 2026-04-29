from __future__ import annotations

import pytest

from rl_task_foundry.config.load import load_config
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.sensitivity import (
    ColumnRef,
    classify_columns,
    resolve_handle_aware_visibility,
)


def test_classify_columns_prefers_qualified_overrides():
    columns = [
        ColumnRef(schema_name="public", table_name="customer", column_name="email"),
        ColumnRef(schema_name="public", table_name="staff", column_name="email"),
        ColumnRef(schema_name="public", table_name="payment", column_name="amount"),
    ]

    classified = classify_columns(
        columns,
        default_visibility="blocked",
        overrides={
            "public.customer.email": "user_visible",
            "staff.email": "internal",
        },
    )

    assert [column.visibility for column in classified] == [
        "user_visible",
        "internal",
        "blocked",
    ]


def test_classify_columns_uses_default_without_name_token_promotion():
    columns = [
        ColumnRef(schema_name="public", table_name="language", column_name="name"),
        ColumnRef(schema_name="public", table_name="film", column_name="title"),
        ColumnRef(schema_name="public", table_name="customer", column_name="name"),
    ]

    classified = classify_columns(
        columns,
        default_visibility="blocked",
        overrides={},
    )

    assert [column.visibility for column in classified] == [
        "blocked",
        "blocked",
        "blocked",
    ]


def test_classify_columns_keeps_all_fields_visible_by_default_without_name_rules():
    columns = [
        ColumnRef(schema_name="public", table_name="film", column_name="title"),
        ColumnRef(schema_name="public", table_name="category", column_name="name"),
        ColumnRef(schema_name="public", table_name="actor", column_name="first_name"),
        ColumnRef(schema_name="public", table_name="customer", column_name="first_name"),
        ColumnRef(schema_name="public", table_name="staff", column_name="name"),
        ColumnRef(schema_name="public", table_name="customer", column_name="email"),
        ColumnRef(schema_name="public", table_name="customer", column_name="api_token"),
    ]

    classified = classify_columns(
        columns,
        default_visibility="user_visible",
        overrides={},
    )

    assert [column.visibility for column in classified] == [
        "user_visible",
        "user_visible",
        "user_visible",
        "user_visible",
        "user_visible",
        "user_visible",
        "user_visible",
    ]


def test_handle_visibility_requires_explicit_override_when_default_is_visible():
    column = ColumnRef(
        schema_name="public",
        table_name="procedureevents",
        column_name="orderid",
    )

    assert (
        resolve_handle_aware_visibility(
            column,
            is_handle=True,
            default_visibility="user_visible",
            overrides={},
        )
        == "blocked"
    )
    assert (
        resolve_handle_aware_visibility(
            column,
            is_handle=True,
            default_visibility="user_visible",
            overrides={"procedureevents.orderid": "user_visible"},
        )
        == "user_visible"
    )
    assert (
        resolve_handle_aware_visibility(
            column,
            is_handle=False,
            default_visibility="user_visible",
            overrides={},
        )
        == "user_visible"
    )


@pytest.mark.asyncio
async def test_postgres_schema_introspector_reads_pagila_schema():
    config = load_config("rl_task_foundry.yaml")
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.visibility.default_visibility,
        visibility_overrides=config.visibility.visibility_overrides,
    )

    graph = await introspector.introspect()

    assert "customer" in graph.table_names()
    assert "payment" in graph.table_names()
    assert "payment_p2022_01" not in graph.table_names()

    customer = graph.get_table("customer", schema_name="public")
    assert customer.primary_key == ("customer_id",)
    assert customer.get_column("customer_id").is_primary_key is True
    assert customer.get_column("customer_id").visibility == "blocked"
    assert customer.get_column("email").visibility == config.visibility.default_visibility

    language = graph.get_table("language", schema_name="public")
    assert language.get_column("name").visibility == "user_visible"

    film_actor = graph.get_table("film_actor", schema_name="public")
    assert film_actor.primary_key == ("actor_id", "film_id")

    payment_customer_edge = next(
        edge
        for edge in graph.edges
        if edge.source_table == "payment"
        and edge.target_table == "customer"
        and edge.source_columns == ("customer_id",)
    )
    assert payment_customer_edge.target_columns == ("customer_id",)
    assert payment_customer_edge.source_is_unique is False

    payment = graph.get_table("payment", schema_name="public")
    assert payment.get_column("customer_id").is_foreign_key is True
    assert payment.get_column("amount").visibility in {"blocked", "internal", "user_visible"}
