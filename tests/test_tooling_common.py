"""Unit tests for tooling/common: schema snapshot, SQL helpers, edges."""

from __future__ import annotations

import asyncio
import json

import pytest

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.tooling.common import (
    EdgeDirection,
    SchemaSnapshot,
    available_edges,
    coerce_param,
    coerce_scalar,
    quote_ident,
    quote_table,
    resolve_edge,
    snapshot_from_dict,
    snapshot_from_graph,
    snapshot_to_dict,
)
from rl_task_foundry.tooling.common.tool_runtime import (
    json_dumps_tool,
    wrap_tool_handler,
)


def _column(
    table: str,
    name: str,
    data_type: str = "integer",
    *,
    schema: str = "public",
    is_primary_key: bool = False,
    is_foreign_key: bool = False,
    visibility: str = "user_visible",
) -> ColumnProfile:
    return ColumnProfile(
        schema_name=schema,
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=1,
        is_nullable=False,
        visibility=visibility,  # type: ignore[arg-type]
        is_primary_key=is_primary_key,
        is_foreign_key=is_foreign_key,
    )


def _toy_graph() -> SchemaGraph:
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[_column("customer", "customer_id", is_primary_key=True)],
        primary_key=("customer_id",),
    )
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            _column("rental", "rental_id", is_primary_key=True),
            _column("rental", "customer_id", is_foreign_key=True),
            _column("rental", "rental_date", data_type="timestamp"),
        ],
        primary_key=("rental_id",),
    )
    edge = ForeignKeyEdge(
        constraint_name="rental_customer_fk",
        source_schema="public",
        source_table="rental",
        source_columns=("customer_id",),
        target_schema="public",
        target_table="customer",
        target_columns=("customer_id",),
    )
    return SchemaGraph(tables=[customer, rental], edges=[edge])


def test_snapshot_from_graph_preserves_columns_and_edges():
    snapshot = snapshot_from_graph(_toy_graph())
    assert isinstance(snapshot, SchemaSnapshot)
    assert snapshot.table_names() == ("customer", "rental")
    rental = snapshot.table("rental")
    assert rental.primary_key == ("rental_id",)
    assert rental.column("customer_id").is_foreign_key is True
    assert snapshot.edges_from("rental")[0].target_table == "customer"
    assert snapshot.edges_to("customer")[0].source_table == "rental"


def test_handle_columns_are_structural_primary_or_foreign_keys():
    table = TableProfile(
        schema_name="public",
        table_name="payment",
        columns=[
            _column("payment", "payment_id", is_primary_key=True),
            _column(
                "payment",
                "payment_date",
                data_type="timestamptz",
                is_primary_key=True,
            ),
            _column(
                "payment",
                "tenant",
                is_primary_key=True,
                visibility="internal",
            ),
            _column("payment", "customer_id", is_foreign_key=True),
        ],
        primary_key=("payment_id", "payment_date", "tenant"),
    )

    snapshot = snapshot_from_graph(SchemaGraph(tables=[table], edges=[]))
    payment = snapshot.table("payment")

    assert payment.column("payment_id").is_handle_column is True
    assert payment.column("customer_id").is_handle_column is True
    assert payment.column("tenant").is_handle_column is True
    assert payment.column("payment_date").is_handle_column is True


def test_snapshot_round_trips_column_visibility():
    graph = _toy_graph()
    customer = graph.get_table("customer", schema_name="public")
    customer.columns.append(
        _column("customer", "api_token", data_type="text", visibility="blocked")
    )

    snapshot = snapshot_from_graph(graph)
    assert snapshot.table("customer").column("api_token").visibility == "blocked"
    assert "api_token" not in snapshot.table("customer").exposed_column_names

    payload = snapshot_to_dict(snapshot)
    restored = snapshot_from_dict(payload)

    restored_column = restored.table("customer").column("api_token")
    assert restored_column.visibility == "blocked"
    assert restored_column.is_exposed is False


def test_snapshot_disambiguates_duplicate_table_names_with_schema_handles():
    public_customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "id", schema="public", is_primary_key=True),
        ],
        primary_key=("id",),
    )
    crm_customer = TableProfile(
        schema_name="crm",
        table_name="customer",
        columns=[
            _column("customer", "id", schema="crm", is_primary_key=True),
            _column(
                "customer",
                "public_customer_id",
                schema="crm",
                is_foreign_key=True,
            ),
        ],
        primary_key=("id",),
    )
    booking = TableProfile(
        schema_name="public",
        table_name="booking",
        columns=[
            _column("booking", "id", is_primary_key=True),
            _column("booking", "customer_id", is_foreign_key=True),
        ],
        primary_key=("id",),
    )
    snapshot = snapshot_from_graph(
        SchemaGraph(
            tables=[public_customer, crm_customer, booking],
            edges=[
                ForeignKeyEdge(
                    constraint_name="crm_customer_public_customer_fk",
                    source_schema="crm",
                    source_table="customer",
                    source_columns=("public_customer_id",),
                    target_schema="public",
                    target_table="customer",
                    target_columns=("id",),
                ),
                ForeignKeyEdge(
                    constraint_name="booking_crm_customer_fk",
                    source_schema="public",
                    source_table="booking",
                    source_columns=("customer_id",),
                    target_schema="crm",
                    target_table="customer",
                    target_columns=("id",),
                ),
            ],
        )
    )

    assert snapshot.table_names() == (
        "public.customer",
        "crm.customer",
        "booking",
    )
    with pytest.raises(KeyError):
        snapshot.table("customer")
    assert snapshot.table("public.customer").handle == "public.customer"
    assert snapshot.table("crm.customer").schema == "crm"
    assert snapshot.table("public.booking").handle == "booking"

    booking_edges = available_edges(snapshot, "booking")
    assert [edge.label for edge in booking_edges] == [
        "booking.customer_id->crm%2Ecustomer"
    ]
    crm_edges = {edge.label for edge in available_edges(snapshot, "crm.customer")}
    assert "crm%2Ecustomer.public_customer_id->public%2Ecustomer" in crm_edges
    assert "crm%2Ecustomer<-booking.customer_id" in crm_edges

    payload = snapshot_to_dict(snapshot)
    restored = snapshot_from_dict(payload)
    assert restored.table_names() == snapshot.table_names()
    assert restored.edges == snapshot.edges


def test_snapshot_preserves_composite_foreign_key_edges():
    order = TableProfile(
        schema_name="public",
        table_name="order",
        columns=[
            _column("order", "tenant_id", is_primary_key=True),
            _column("order", "order_id", is_primary_key=True),
        ],
        primary_key=("tenant_id", "order_id"),
    )
    line_item = TableProfile(
        schema_name="public",
        table_name="line_item",
        columns=[
            _column("line_item", "tenant_id", is_foreign_key=True),
            _column("line_item", "order_id", is_foreign_key=True),
            _column("line_item", "line_no", is_primary_key=True),
        ],
        primary_key=("tenant_id", "order_id", "line_no"),
    )
    snapshot = snapshot_from_graph(
        SchemaGraph(
            tables=[order, line_item],
            edges=[
                ForeignKeyEdge(
                    constraint_name="line_item_order_fk",
                    source_schema="public",
                    source_table="line_item",
                    source_columns=("tenant_id", "order_id"),
                    target_schema="public",
                    target_table="order",
                    target_columns=("tenant_id", "order_id"),
                )
            ],
        )
    )

    edge = snapshot.edges[0]
    assert edge.source_columns == ("tenant_id", "order_id")
    assert edge.target_columns == ("tenant_id", "order_id")
    assert edge.forward_label == (
        "line_item.(tenant_id,order_id)->order.(tenant_id,order_id)"
    )
    assert [
        typed.label for typed in available_edges(snapshot, "line_item")
    ] == ["line_item.(tenant_id,order_id)->order.(tenant_id,order_id)"]
    assert [
        typed.label for typed in available_edges(snapshot, "order")
    ] == ["order.(tenant_id,order_id)<-line_item.(tenant_id,order_id)"]

    restored = snapshot_from_dict(snapshot_to_dict(snapshot))
    assert restored.edges == snapshot.edges


def test_relation_labels_encode_delimiter_characters_without_collisions():
    source = TableProfile(
        schema_name="public",
        table_name="src",
        columns=[
            _column("src", "id", is_primary_key=True),
            _column("src", "x->y", is_foreign_key=True),
            _column("src", "x", is_foreign_key=True),
        ],
        primary_key=("id",),
    )
    target_plain = TableProfile(
        schema_name="public",
        table_name="z",
        columns=[_column("z", "id", is_primary_key=True)],
        primary_key=("id",),
    )
    target_with_arrow = TableProfile(
        schema_name="public",
        table_name="y->z",
        columns=[_column("y->z", "id", is_primary_key=True)],
        primary_key=("id",),
    )

    snapshot = snapshot_from_graph(
        SchemaGraph(
            tables=[source, target_plain, target_with_arrow],
            edges=[
                ForeignKeyEdge(
                    constraint_name="src_arrow_col_fk",
                    source_schema="public",
                    source_table="src",
                    source_columns=("x->y",),
                    target_schema="public",
                    target_table="z",
                    target_columns=("id",),
                ),
                ForeignKeyEdge(
                    constraint_name="src_arrow_table_fk",
                    source_schema="public",
                    source_table="src",
                    source_columns=("x",),
                    target_schema="public",
                    target_table="y->z",
                    target_columns=("id",),
                ),
            ],
        )
    )

    labels = [edge.label for edge in available_edges(snapshot, "src")]

    assert labels == [
        "src.x%2D%3Ey->z",
        "src.x->y%2D%3Ez",
    ]
    assert len(labels) == len(set(labels))


def test_snapshot_from_graph_rejects_mismatched_fk_column_counts():
    parent = TableProfile(
        schema_name="public",
        table_name="parent",
        columns=[
            _column("parent", "id", is_primary_key=True),
            _column("parent", "tenant_id"),
        ],
        primary_key=("id",),
    )
    child = TableProfile(
        schema_name="public",
        table_name="child",
        columns=[
            _column("child", "id", is_primary_key=True),
            _column("child", "parent_id", is_foreign_key=True),
        ],
        primary_key=("id",),
    )

    with pytest.raises(ValueError, match="mismatched"):
        snapshot_from_graph(
            SchemaGraph(
                tables=[parent, child],
                edges=[
                    ForeignKeyEdge(
                        constraint_name="bad_fk",
                        source_schema="public",
                        source_table="child",
                        source_columns=("parent_id",),
                        target_schema="public",
                        target_table="parent",
                        target_columns=("tenant_id", "id"),
                    )
                ],
            )
        )


def test_snapshot_does_not_let_dotted_table_names_shadow_qualified_aliases():
    dotted_table = TableProfile(
        schema_name="public",
        table_name="crm.customer",
        columns=[
            _column("crm.customer", "id", is_primary_key=True),
        ],
        primary_key=("id",),
    )
    crm_customer = TableProfile(
        schema_name="crm",
        table_name="customer",
        columns=[
            _column("customer", "id", schema="crm", is_primary_key=True),
        ],
        primary_key=("id",),
    )

    snapshot = snapshot_from_graph(
        SchemaGraph(tables=[dotted_table, crm_customer], edges=[])
    )

    assert snapshot.table_names() == ("public.crm.customer", "customer")
    assert snapshot.table("crm.customer").schema == "crm"
    assert snapshot.table("public.crm.customer").name == "crm.customer"


def test_snapshot_table_lookup_prefers_exact_handle_over_ambiguous_qualified_name():
    dotted_schema_table = TableProfile(
        schema_name="public.crm",
        table_name="customer",
        columns=[
            _column("customer", "id", schema="public.crm", is_primary_key=True),
        ],
        primary_key=("id",),
    )
    dotted_name_table = TableProfile(
        schema_name="public",
        table_name="crm.customer",
        columns=[
            _column("crm.customer", "id", is_primary_key=True),
        ],
        primary_key=("id",),
    )

    snapshot = snapshot_from_graph(
        SchemaGraph(tables=[dotted_schema_table, dotted_name_table], edges=[])
    )

    assert snapshot.table_names() == ("customer", "public.crm.customer")
    assert snapshot.table("customer").schema == "public.crm"
    assert snapshot.table("public.crm.customer").schema == "public"


def test_snapshot_from_dict_rejects_duplicate_table_handles():
    payload = snapshot_to_dict(snapshot_from_graph(_toy_graph()))
    tables = payload["tables"]
    assert isinstance(tables, list)
    rental = tables[1]
    assert isinstance(rental, dict)
    rental["handle"] = "customer"

    with pytest.raises(ValueError, match="handles must be unique"):
        snapshot_from_dict(payload)


def test_available_edges_lists_both_directions():
    snapshot = snapshot_from_graph(_toy_graph())
    rental_edges = available_edges(snapshot, "rental")
    customer_edges = available_edges(snapshot, "customer")
    assert any(edge.direction is EdgeDirection.FORWARD for edge in rental_edges)
    assert any(
        edge.direction is EdgeDirection.REVERSE for edge in customer_edges
    )
    forward = next(
        edge
        for edge in rental_edges
        if edge.direction is EdgeDirection.FORWARD
    )
    assert forward.origin_table == "rental"
    assert forward.destination_table == "customer"
    reverse = next(
        edge
        for edge in customer_edges
        if edge.direction is EdgeDirection.REVERSE
    )
    assert reverse.origin_table == "customer"
    assert reverse.destination_table == "rental"


def test_resolve_edge_by_label_round_trips():
    snapshot = snapshot_from_graph(_toy_graph())
    edges = available_edges(snapshot, "rental")
    forward = next(
        edge for edge in edges if edge.direction is EdgeDirection.FORWARD
    )
    resolved = resolve_edge(snapshot, "rental", forward.label)
    assert resolved == forward
    with pytest.raises(KeyError):
        resolve_edge(snapshot, "rental", "nonexistent")


def test_sql_identifier_quoting_handles_embedded_quotes():
    assert quote_ident("simple") == '"simple"'
    assert quote_ident('quo"te') == '"quo""te"'
    assert quote_table("public", "rental") == '"public"."rental"'


def test_coerce_param_allows_scalars_and_lists_and_rejects_others():
    assert coerce_param(None) is None
    assert coerce_param(42) == 42
    assert coerce_param("text") == "text"
    assert coerce_param([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(TypeError):
        coerce_param({"bad": 1})
    with pytest.raises(TypeError):
        coerce_param((1, 2))


def test_coerce_scalar_promotes_integer_pk_from_string():
    # LLM tool payloads deliver PK values as JSON strings; asyncpg's binary
    # protocol rejects strings on integer columns, so the tool factory must
    # coerce before binding.
    assert coerce_scalar("5244", "integer") == 5244
    assert coerce_scalar("9999", "bigint") == 9999
    assert coerce_scalar("7", "smallint") == 7
    assert coerce_scalar(123, "integer") == 123  # int passes through
    assert coerce_scalar(["1", "2", "3"], "integer") == [1, 2, 3]
    # schema introspection stores Postgres udt_name (int2/int4/int8,
    # float4/float8) rather than the information_schema data_type label.
    assert coerce_scalar("5244", "int4") == 5244
    assert coerce_scalar("9999", "int8") == 9999
    assert coerce_scalar("7", "int2") == 7
    assert coerce_scalar("3.14", "float4") == pytest.approx(3.14)
    assert coerce_scalar("2.71828", "float8") == pytest.approx(2.71828)


def test_coerce_scalar_covers_numeric_and_boolean_columns():
    import datetime as _dt
    from decimal import Decimal

    assert coerce_scalar("3.14", "real") == 3.14
    assert coerce_scalar("2.71828", "double precision") == pytest.approx(2.71828)
    assert coerce_scalar("10.50", "numeric") == Decimal("10.50")
    assert coerce_scalar(4.99, "numeric") == Decimal("4.99")
    assert coerce_scalar([4.99, "2.99"], "numeric") == [
        Decimal("4.99"),
        Decimal("2.99"),
    ]
    assert coerce_param(Decimal("4.99")) == Decimal("4.99")
    assert coerce_param([Decimal("4.99")]) == [Decimal("4.99")]
    assert coerce_scalar("true", "boolean") is True
    assert coerce_scalar("no", "boolean") is False
    assert coerce_scalar("2026-04-18", "date") == _dt.date(2026, 4, 18)
    assert coerce_scalar("2026-04-18T10:00:00", "timestamp") == _dt.datetime(
        2026, 4, 18, 10, 0, 0
    )
    assert coerce_scalar("2022-07-01", "timestamptz") == _dt.datetime(
        2022, 7, 1, tzinfo=_dt.UTC
    )
    assert coerce_scalar(
        "2022-07-01T09:00:00+09:00", "timestamp with time zone"
    ) == _dt.datetime(2022, 7, 1, 9, 0, tzinfo=_dt.timezone(_dt.timedelta(hours=9)))
    assert coerce_scalar(
        "2022-07-01T09:00:00+09:00", "timestamp"
    ) == _dt.datetime(2022, 7, 1)


def test_coerce_scalar_leaves_unknown_types_and_non_strings_alone():
    assert coerce_scalar("hello", "text") == "hello"
    assert coerce_scalar(42, "integer") == 42
    assert coerce_scalar(None, "integer") is None
    # Unknown / unmapped data_type: passthrough.
    assert coerce_scalar("some-uuid-value", "uuid") == "some-uuid-value"


def test_tool_json_serializes_temporal_values_as_iso_strings():
    import datetime as _dt
    from decimal import Decimal

    payload = {
        "created_at": _dt.datetime(2022, 7, 23, 9, 13, 13, tzinfo=_dt.UTC),
        "created_on": _dt.date(2022, 7, 23),
        "created_time": _dt.time(9, 13, 13),
        "amount": Decimal("4.99"),
    }

    parsed = json.loads(json_dumps_tool(payload))

    assert parsed == {
        "created_at": "2022-07-23T09:13:13+00:00",
        "created_on": "2022-07-23",
        "created_time": "09:13:13",
        "amount": "4.99",
    }


@pytest.mark.asyncio
async def test_tool_runtime_lock_serializes_handler_calls():
    lock = asyncio.Lock()
    active = False

    async def handler(payload):
        nonlocal active
        if active:
            raise RuntimeError("concurrent handler use")
        active = True
        try:
            await asyncio.sleep(0)
            return {"ok": True, "value": payload["value"]}
        finally:
            active = False

    invoke = wrap_tool_handler(handler, lock=lock)

    first, second = await asyncio.gather(
        invoke(None, '{"value": 1}'),
        invoke(None, '{"value": 2}'),
    )

    assert json.loads(first) == {"ok": True, "value": 1}
    assert json.loads(second) == {"ok": True, "value": 2}
