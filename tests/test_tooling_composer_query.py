"""Tests for the alias-qualified composer query DSL.

Unit tests exercise strict parsing, validation, and SQL shape with a
recording stub connection. Integration tests hit live pagila to confirm the
DSL authors canonical answers through real joins and typed predicates.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import asyncpg
import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.infra.db import (
    _apply_session_settings,
    solver_session_settings,
)
from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.tooling.common import snapshot_from_graph
from rl_task_foundry.tooling.composer import ComposerSession, query
from rl_task_foundry.tooling.composer._session import _Row


class _RecordingConnection:
    def __init__(
        self,
        rows: list[dict[str, object]] | None = None,
        *,
        row_batches: list[list[dict[str, object]]] | None = None,
    ) -> None:
        self.rows: list[dict[str, object]] = rows or []
        self.row_batches = row_batches
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> Sequence[_Row]:
        self.calls.append((sql, args))
        if self.row_batches is not None:
            batch_index = min(len(self.calls) - 1, len(self.row_batches) - 1)
            return cast(Sequence[_Row], list(self.row_batches[batch_index]))
        return cast(Sequence[_Row], list(self.rows))

    async def fetchrow(self, sql: str, *args: object):  # pragma: no cover
        raise AssertionError("fetchrow must not be used by query")

    async def fetchval(self, sql: str, *args: object):  # pragma: no cover
        raise AssertionError("fetchval must not be used by query")


def _column(
    table: str,
    name: str,
    data_type: str = "int4",
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


def _snapshot():
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "customer_id", is_primary_key=True),
            _column("customer", "store_id"),
            _column("customer", "first_name", data_type="text"),
            _column("customer", "email", data_type="text", visibility="internal"),
            _column("customer", "api_token", data_type="text", visibility="blocked"),
        ],
        primary_key=("customer_id",),
    )
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            _column("rental", "rental_id", is_primary_key=True),
            _column("rental", "customer_id", is_foreign_key=True),
            _column("rental", "inventory_id", is_foreign_key=True),
            _column("rental", "rental_date", data_type="timestamp"),
        ],
        primary_key=("rental_id",),
    )
    inventory = TableProfile(
        schema_name="public",
        table_name="inventory",
        columns=[
            _column("inventory", "inventory_id", is_primary_key=True),
            _column("inventory", "film_id", is_foreign_key=True),
        ],
        primary_key=("inventory_id",),
    )
    film = TableProfile(
        schema_name="public",
        table_name="film",
        columns=[
            _column("film", "film_id", is_primary_key=True),
            _column("film", "title", data_type="text"),
        ],
        primary_key=("film_id",),
    )
    edges = [
        ForeignKeyEdge(
            constraint_name="rental_customer",
            source_schema="public",
            source_table="rental",
            source_columns=("customer_id",),
            target_schema="public",
            target_table="customer",
            target_columns=("customer_id",),
        ),
        ForeignKeyEdge(
            constraint_name="rental_inventory",
            source_schema="public",
            source_table="rental",
            source_columns=("inventory_id",),
            target_schema="public",
            target_table="inventory",
            target_columns=("inventory_id",),
        ),
        ForeignKeyEdge(
            constraint_name="inventory_film",
            source_schema="public",
            source_table="inventory",
            source_columns=("film_id",),
            target_schema="public",
            target_table="film",
            target_columns=("film_id",),
        ),
    ]
    return snapshot_from_graph(
        SchemaGraph(
            tables=[customer, rental, inventory, film],
            edges=edges,
        )
    )


def _duplicate_name_snapshot():
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
            _column("customer", "name", data_type="text", schema="crm"),
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
    return snapshot_from_graph(
        SchemaGraph(
            tables=[public_customer, crm_customer, booking],
            edges=[
                ForeignKeyEdge(
                    constraint_name="booking_crm_customer_fk",
                    source_schema="public",
                    source_table="booking",
                    source_columns=("customer_id",),
                    target_schema="crm",
                    target_table="customer",
                    target_columns=("id",),
                )
            ],
        )
    )


def _composite_order_snapshot():
    order = TableProfile(
        schema_name="public",
        table_name="order",
        columns=[
            _column("order", "tenant_id", is_primary_key=True),
            _column("order", "order_id", is_primary_key=True),
            _column("order", "status", data_type="text"),
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
            _column("line_item", "sku", data_type="text"),
        ],
        primary_key=("tenant_id", "order_id", "line_no"),
    )
    return snapshot_from_graph(
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


def _stub_session(
    rows: list[dict[str, object]] | None = None,
) -> tuple[ComposerSession, _RecordingConnection]:
    conn = _RecordingConnection(rows=rows)
    session = ComposerSession(snapshot=_snapshot(), connection=conn)
    return session, conn


def _stub_session_with_batches(
    row_batches: list[list[dict[str, object]]],
) -> tuple[ComposerSession, _RecordingConnection]:
    conn = _RecordingConnection(row_batches=row_batches)
    session = ComposerSession(snapshot=_snapshot(), connection=conn)
    return session, conn


def _from(table: str, alias: str) -> dict[str, str]:
    return {"table": table, "as": alias}


def _ref(alias: str, column: str) -> dict[str, str]:
    return {"as": alias, "column": column}


def _select(alias: str, column: str, output: str | None = None) -> dict[str, object]:
    return {"ref": _ref(alias, column), "as": output or column}


def _group(alias: str, column: str, output: str | None = None) -> dict[str, object]:
    return {"ref": _ref(alias, column), "as": output or column}


def _agg(
    fn: str,
    output: str,
    *,
    alias: str | None = None,
    column: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"fn": fn, "as": output}
    if alias is not None and column is not None:
        payload["ref"] = _ref(alias, column)
    return payload


def _order_ref(
    alias: str,
    column: str,
    direction: str = "asc",
) -> dict[str, object]:
    return {"ref": _ref(alias, column), "direction": direction}


def _order_output(output: str, direction: str = "asc") -> dict[str, object]:
    return {"output": output, "direction": direction}


# ---------- basic SELECT shapes ----------


@pytest.mark.asyncio
async def test_query_from_only_selects_all_terminal_columns():
    session, conn = _stub_session()
    await query(session, spec={"from": _from("customer", "c")})
    sql, params = conn.calls[0]
    assert "SELECT t0.\"customer_id\" AS \"customer_id\"" in sql
    assert "t0.\"first_name\" AS \"first_name\"" in sql
    assert "api_token" not in sql
    assert "FROM \"public\".\"customer\" AS t0" in sql
    assert params == ()


@pytest.mark.asyncio
async def test_query_select_uses_alias_qualified_refs_and_output_names():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [
                _select("c", "customer_id", "customer_pk"),
                _select("c", "first_name"),
            ],
        },
    )
    sql, _ = conn.calls[0]
    assert "SELECT t0.\"customer_id\" AS \"customer_pk\"" in sql
    assert "t0.\"first_name\" AS \"first_name\"" in sql
    assert "store_id" not in sql


@pytest.mark.asyncio
async def test_query_returns_visibility_provenance_for_outputs_and_refs():
    session, _ = _stub_session()

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "where": [
                {"ref": _ref("c", "email"), "op": "is_not_null"},
            ],
            "select": [_select("c", "email", "contact")],
            "order_by": [_order_ref("c", "email", "asc")],
            "limit": 1,
        },
    )

    assert result["column_sources"] == [
        {
            "output": "contact",
            "kind": "select",
            "table": "customer",
            "column": "email",
            "visibility": "internal",
            "is_handle": False,
            "is_primary_key": False,
            "table_primary_key": ["customer_id"],
            "table_has_primary_key": True,
            "value_exposes_source": True,
        }
    ]
    assert result["referenced_columns"] == [
        {
            "usage": "where",
            "table": "customer",
            "column": "email",
            "visibility": "internal",
            "is_handle": False,
            "is_primary_key": False,
            "table_primary_key": ["customer_id"],
            "op": "is_not_null",
            "value": None,
        },
        {
            "usage": "order_by",
            "table": "customer",
            "column": "email",
            "visibility": "internal",
            "is_handle": False,
            "is_primary_key": False,
            "table_primary_key": ["customer_id"],
            "direction": "asc",
        },
    ]


@pytest.mark.asyncio
async def test_query_marks_label_sources_without_primary_key():
    detail = TableProfile(
        schema_name="public",
        table_name="event_detail",
        columns=[
            _column("event_detail", "event_id", is_foreign_key=True),
            _column("event_detail", "detail_text", data_type="text"),
        ],
        primary_key=(),
    )
    session = ComposerSession(
        snapshot=snapshot_from_graph(SchemaGraph(tables=[detail], edges=[])),
        connection=_RecordingConnection(),
    )

    result = await query(
        session,
        spec={
            "from": _from("event_detail", "d"),
            "select": [_select("d", "detail_text")],
        },
    )

    assert result["column_sources"] == [
        {
            "output": "detail_text",
            "kind": "select",
            "table": "event_detail",
            "column": "detail_text",
            "visibility": "user_visible",
            "is_handle": False,
            "is_primary_key": False,
            "table_primary_key": [],
            "table_has_primary_key": False,
            "value_exposes_source": True,
        }
    ]


@pytest.mark.asyncio
async def test_query_where_binds_and_coerces_params():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "where": [
                {"ref": _ref("c", "store_id"), "op": "eq", "value": "1"},
                {
                    "ref": _ref("c", "customer_id"),
                    "op": "in",
                    "value": ["1", "2", "3"],
                },
            ],
            "select": [_select("c", "customer_id")],
        },
    )
    sql, params = conn.calls[0]
    assert "WHERE t0.\"store_id\" = $1 AND t0.\"customer_id\" = ANY($2::int4[])" in sql
    assert params == (1, [1, 2, 3])


@pytest.mark.asyncio
async def test_query_where_supports_neq_and_nullary_ops():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "where": [
                {"ref": _ref("c", "store_id"), "op": "neq", "value": 1},
                {"ref": _ref("c", "first_name"), "op": "is_not_null"},
            ],
            "select": [_select("c", "customer_id")],
        },
    )
    sql, params = conn.calls[0]
    assert (
        "WHERE t0.\"store_id\" <> $1 AND t0.\"first_name\" IS NOT NULL"
        in sql
    )
    assert params == (1,)


@pytest.mark.asyncio
async def test_query_where_rejects_null_for_binary_ops():
    session, _ = _stub_session()
    with pytest.raises(TypeError, match="non-null"):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "where": [
                    {"ref": _ref("c", "store_id"), "op": "eq", "value": None},
                ],
                "select": [_select("c", "customer_id")],
            },
        )


@pytest.mark.asyncio
async def test_query_limit_and_order_by_ref_emitted_in_order():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [_select("c", "customer_id")],
            "order_by": [_order_ref("c", "customer_id", "desc")],
            "limit": 3,
        },
    )
    sql, _ = conn.calls[0]
    assert "ORDER BY t0.\"customer_id\" DESC" in sql
    assert sql.rstrip().endswith("LIMIT 3")


@pytest.mark.asyncio
async def test_query_reports_duplicate_limited_order_key_diagnostics():
    session, _ = _stub_session(
        rows=[
            {"customer_id": 1, "first_name": "ALICE"},
            {"customer_id": 2, "first_name": "ALICE"},
            {"customer_id": 3, "first_name": "BOB"},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [
                _select("c", "customer_id"),
                _select("c", "first_name"),
            ],
            "order_by": [_order_ref("c", "first_name", "asc")],
            "limit": 3,
        },
    )

    assert result["ordering_diagnostics"] == {
        "order_by_outputs": ["first_name"],
        "duplicate_order_key_in_returned_rows": True,
        "returned_row_count": 3,
        "limit": 3,
    }


@pytest.mark.asyncio
async def test_query_reports_limit_boundary_tie_diagnostics():
    session, conn = _stub_session_with_batches(
        row_batches=[
            [
                {"store_id": 1, "first_name": "ALICE"},
                {"store_id": 1, "first_name": "BOB"},
                {"store_id": 1, "first_name": "CHRIS"},
            ],
            [
                {"store_id": 1, "first_name": "ALICE"},
                {"store_id": 1, "first_name": "BOB"},
                {"store_id": 1, "first_name": "CHRIS"},
                {"store_id": 2, "first_name": "CHRIS"},
            ],
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [
                _select("c", "store_id"),
                _select("c", "first_name"),
            ],
            "order_by": [_order_ref("c", "first_name", "asc")],
            "limit": 3,
        },
    )

    assert result["rows"] == [
        {"store_id": 1, "first_name": "ALICE"},
        {"store_id": 1, "first_name": "BOB"},
        {"store_id": 1, "first_name": "CHRIS"},
    ]
    assert result["ordering_diagnostics"] == {
        "order_by_outputs": ["first_name"],
        "duplicate_order_key_in_returned_rows": False,
        "returned_row_count": 3,
        "limit": 3,
        "limit_boundary_tie": True,
    }
    assert len(conn.calls) == 2
    diagnostic_sql, _ = conn.calls[1]
    assert diagnostic_sql.rstrip().endswith("LIMIT 4")


@pytest.mark.asyncio
async def test_query_reports_unrepresented_order_by_tie_breaker_diagnostics():
    session, _ = _stub_session(
        rows=[
            {"first_name": "ALICE", "store_id": 1, "__rtf_order_1": 2},
            {"first_name": "ALICE", "store_id": 2, "__rtf_order_1": 1},
            {"first_name": "BOB", "store_id": 1, "__rtf_order_1": 3},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [
                _select("c", "first_name"),
                _select("c", "store_id"),
            ],
            "order_by": [
                _order_ref("c", "first_name", "asc"),
                _order_ref("c", "customer_id", "desc"),
            ],
            "limit": 3,
        },
    )

    assert result["ordering_diagnostics"] == {
        "order_by_outputs": ["first_name"],
        "unrepresented_order_by_tie_breakers": [
            {
                "table": "customer",
                "column": "customer_id",
                "direction": "desc",
                "is_handle": True,
            }
        ],
        "handle_order_by_columns": [
            {
                "table": "customer",
                "column": "customer_id",
                "direction": "desc",
                "is_selected_output": False,
            }
        ],
        "returned_row_count": 3,
        "limit": 3,
    }


@pytest.mark.asyncio
async def test_query_reports_unrepresented_order_by_tie_breaker_without_limit():
    session, _ = _stub_session(
        rows=[
            {"first_name": "ALICE", "store_id": 1, "__rtf_order_1": 2},
            {"first_name": "ALICE", "store_id": 2, "__rtf_order_1": 1},
            {"first_name": "BOB", "store_id": 1, "__rtf_order_1": 3},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [
                _select("c", "first_name"),
                _select("c", "store_id"),
            ],
            "order_by": [
                _order_ref("c", "first_name", "asc"),
                _order_ref("c", "customer_id", "desc"),
            ],
        },
    )

    assert result["ordering_diagnostics"] == {
        "order_by_outputs": ["first_name"],
        "unrepresented_order_by_tie_breakers": [
            {
                "table": "customer",
                "column": "customer_id",
                "direction": "desc",
                "is_handle": True,
            }
        ],
        "handle_order_by_columns": [
            {
                "table": "customer",
                "column": "customer_id",
                "direction": "desc",
                "is_selected_output": False,
            }
        ],
        "returned_row_count": 3,
    }


@pytest.mark.asyncio
async def test_query_reports_duplicate_order_key_without_limit():
    session, _ = _stub_session(
        rows=[
            {"customer_id": 1, "first_name": "ALICE"},
            {"customer_id": 2, "first_name": "ALICE"},
            {"customer_id": 3, "first_name": "BOB"},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [
                _select("c", "customer_id"),
                _select("c", "first_name"),
            ],
            "order_by": [_order_ref("c", "first_name", "asc")],
        },
    )

    assert result["ordering_diagnostics"] == {
        "order_by_outputs": ["first_name"],
        "duplicate_order_key_in_returned_rows": True,
        "returned_row_count": 3,
    }


@pytest.mark.asyncio
async def test_query_does_not_reject_unrepresented_visible_tie_breaker():
    session, _ = _stub_session(
        rows=[
            {"first_name": "ALICE", "__rtf_order_1": 1},
            {"first_name": "ALICE", "__rtf_order_1": 2},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [_select("c", "first_name")],
            "order_by": [
                _order_ref("c", "first_name", "asc"),
                _order_ref("c", "store_id", "desc"),
            ],
            "limit": 2,
        },
    )

    assert "ordering_diagnostics" not in result


@pytest.mark.asyncio
async def test_query_reports_duplicate_full_order_key_with_visible_tie_breaker():
    session, _ = _stub_session(
        rows=[
            {"first_name": "ALICE", "customer_id": 1, "__rtf_order_1": 7},
            {"first_name": "ALICE", "customer_id": 2, "__rtf_order_1": 7},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [
                _select("c", "first_name"),
                _select("c", "customer_id"),
            ],
            "order_by": [
                _order_ref("c", "first_name", "asc"),
                _order_ref("c", "store_id", "desc"),
            ],
            "limit": 2,
        },
    )

    assert result["ordering_diagnostics"] == {
        "order_by_outputs": ["first_name"],
        "duplicate_order_key_in_returned_rows": True,
        "returned_row_count": 2,
        "limit": 2,
    }


@pytest.mark.asyncio
async def test_query_does_not_reject_hidden_tie_breaker_for_identical_answers():
    session, _ = _stub_session(
        rows=[
            {"first_name": "ALICE", "__rtf_order_1": 2},
            {"first_name": "ALICE", "__rtf_order_1": 1},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [_select("c", "first_name")],
            "order_by": [
                _order_ref("c", "first_name", "asc"),
                _order_ref("c", "customer_id", "desc"),
            ],
            "limit": 2,
        },
    )

    assert "ordering_diagnostics" not in result


@pytest.mark.asyncio
async def test_query_reports_duplicate_projected_answer_rows():
    session, _ = _stub_session(
        rows=[
            {"first_name": "ALICE"},
            {"first_name": "ALICE"},
            {"first_name": "BOB"},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [_select("c", "first_name")],
        },
    )

    assert result["projection_diagnostics"] == {
        "duplicate_answer_rows": True,
        "duplicate_answer_row_groups": [[0, 1]],
        "unique_answer_row_count": 2,
        "returned_row_count": 3,
    }


@pytest.mark.asyncio
async def test_query_reports_missing_order_by_for_limited_multirow_list():
    session, _ = _stub_session(
        rows=[
            {"customer_id": 1},
            {"customer_id": 2},
        ]
    )

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [_select("c", "customer_id")],
            "limit": 2,
        },
    )

    assert result["ordering_diagnostics"] == {
        "missing_order_by_for_limit": True,
        "returned_row_count": 2,
        "limit": 2,
    }


# ---------- join chain ----------


@pytest.mark.asyncio
async def test_query_single_forward_join_moves_target_to_destination():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("rental", "r"),
            "join": [{"via_edge": "rental.customer_id->customer", "as": "c"}],
            "select": [_select("c", "first_name")],
        },
    )
    sql, _ = conn.calls[0]
    assert (
        "JOIN \"public\".\"customer\" AS t1 "
        "ON t1.\"customer_id\" = t0.\"customer_id\""
    ) in sql
    assert "t1.\"first_name\" AS \"first_name\"" in sql


@pytest.mark.asyncio
async def test_query_multi_join_walks_rental_to_film():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("rental", "r"),
            "join": [
                {"via_edge": "rental.inventory_id->inventory", "as": "i"},
                {"via_edge": "inventory.film_id->film", "as": "f"},
            ],
            "select": [_select("f", "title")],
        },
    )
    sql, _ = conn.calls[0]
    assert (
        "JOIN \"public\".\"inventory\" AS t1 "
        "ON t1.\"inventory_id\" = t0.\"inventory_id\""
    ) in sql
    assert (
        "JOIN \"public\".\"film\" AS t2 "
        "ON t2.\"film_id\" = t1.\"film_id\""
    ) in sql
    assert "t2.\"title\" AS \"title\"" in sql


@pytest.mark.asyncio
async def test_query_join_can_branch_from_prior_aliases():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("rental", "r"),
            "join": [
                {
                    "from": "r",
                    "via_edge": "rental.inventory_id->inventory",
                    "as": "i",
                },
                {
                    "from": "r",
                    "via_edge": "rental.customer_id->customer",
                    "as": "c",
                },
            ],
            "select": [
                _select("i", "inventory_id"),
                _select("c", "customer_id"),
            ],
        },
    )

    sql, _ = conn.calls[0]
    assert (
        "JOIN \"public\".\"inventory\" AS t1 "
        "ON t1.\"inventory_id\" = t0.\"inventory_id\""
    ) in sql
    assert (
        "JOIN \"public\".\"customer\" AS t2 "
        "ON t2.\"customer_id\" = t0.\"customer_id\""
    ) in sql
    assert "t1.\"inventory_id\" AS \"inventory_id\"" in sql
    assert "t2.\"customer_id\" AS \"customer_id\"" in sql


@pytest.mark.asyncio
async def test_query_where_can_filter_joined_tables():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("film", "f"),
            "join": [
                {"via_edge": "film<-inventory.film_id", "as": "i"},
                {"via_edge": "inventory<-rental.inventory_id", "as": "r"},
            ],
            "where": [
                {"ref": _ref("f", "film_id"), "op": "eq", "value": "128"},
                {
                    "ref": _ref("r", "rental_date"),
                    "op": "gte",
                    "value": "2005-07-01T00:00:00",
                },
            ],
            "select": [_select("r", "rental_date")],
            "order_by": [_order_ref("r", "rental_date", "asc")],
            "limit": 5,
        },
    )
    sql, params = conn.calls[0]
    assert "WHERE t0.\"film_id\" = $1 AND t2.\"rental_date\" >= $2" in sql
    assert "ORDER BY t2.\"rental_date\" ASC" in sql
    assert params[0] == 128
    assert isinstance(params[1], dt.datetime)


@pytest.mark.asyncio
async def test_query_select_spans_from_and_joined_tables():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("inventory", "i"),
            "join": [
                {"via_edge": "inventory<-rental.inventory_id", "as": "r"},
                {"via_edge": "rental.customer_id->customer", "as": "c"},
            ],
            "select": [
                _select("r", "rental_date"),
                _select("c", "first_name"),
            ],
        },
    )
    sql, _ = conn.calls[0]
    assert "t1.\"rental_date\" AS \"rental_date\"" in sql
    assert "t2.\"first_name\" AS \"first_name\"" in sql


@pytest.mark.asyncio
async def test_query_reverse_join_follows_edge_backwards():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "join": [{"via_edge": "customer<-rental.customer_id", "as": "r"}],
            "select": [_select("r", "rental_date")],
        },
    )
    sql, _ = conn.calls[0]
    assert (
        "JOIN \"public\".\"rental\" AS t1 "
        "ON t1.\"customer_id\" = t0.\"customer_id\""
    ) in sql


@pytest.mark.asyncio
async def test_query_join_chain_uses_schema_handles_for_duplicate_tables():
    conn = _RecordingConnection()
    session = ComposerSession(
        snapshot=_duplicate_name_snapshot(),
        connection=conn,
    )

    result = await query(
        session,
        spec={
            "from": _from("booking", "b"),
            "join": [
                {"via_edge": "booking.customer_id->crm%2Ecustomer", "as": "c"},
            ],
            "select": [_select("c", "name")],
        },
    )

    sql, _ = conn.calls[0]
    assert "JOIN \"crm\".\"customer\" AS t1" in sql
    assert "ON t1.\"id\" = t0.\"customer_id\"" in sql
    assert result["column_sources"] == [
        {
            "output": "name",
            "kind": "select",
            "table": "crm.customer",
            "column": "name",
            "visibility": "user_visible",
            "is_handle": False,
            "is_primary_key": False,
            "table_primary_key": ["id"],
            "table_has_primary_key": True,
            "value_exposes_source": True,
        }
    ]


@pytest.mark.asyncio
async def test_query_join_chain_supports_composite_fk_edges():
    conn = _RecordingConnection()
    session = ComposerSession(
        snapshot=_composite_order_snapshot(),
        connection=conn,
    )

    await query(
        session,
        spec={
            "from": _from("line_item", "li"),
            "join": [
                {
                    "via_edge": (
                        "line_item.(tenant_id,order_id)->"
                        "order.(tenant_id,order_id)"
                    ),
                    "as": "o",
                },
            ],
            "select": [_select("o", "status")],
        },
    )

    sql, _ = conn.calls[0]
    assert (
        "ON t1.\"tenant_id\" = t0.\"tenant_id\" "
        "AND t1.\"order_id\" = t0.\"order_id\""
    ) in sql


# ---------- aggregate + group_by ----------


@pytest.mark.asyncio
async def test_query_aggregate_count_without_group_by():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("rental", "r"),
            "aggregate": [_agg("count", "total")],
        },
    )
    sql, _ = conn.calls[0]
    assert "COUNT(*) AS \"total\"" in sql
    assert "GROUP BY" not in sql


@pytest.mark.asyncio
async def test_query_aggregate_with_group_by_emits_group_and_order_by_output():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("rental", "r"),
            "group_by": [_group("r", "customer_id")],
            "aggregate": [
                _agg("count", "rentals"),
                _agg("max", "last", alias="r", column="rental_date"),
            ],
            "order_by": [_order_output("rentals", "desc")],
            "limit": 3,
        },
    )
    sql, _ = conn.calls[0]
    assert (
        "SELECT t0.\"customer_id\" AS \"customer_id\", "
        "COUNT(*) AS \"rentals\", "
        "MAX(t0.\"rental_date\") AS \"last\""
    ) in sql
    assert "GROUP BY t0.\"customer_id\"" in sql
    assert "ORDER BY \"rentals\" DESC" in sql
    assert sql.rstrip().endswith("LIMIT 3")


@pytest.mark.asyncio
async def test_query_order_by_output_reports_source_column_provenance():
    session, _ = _stub_session()

    result = await query(
        session,
        spec={
            "from": _from("customer", "c"),
            "select": [_select("c", "first_name", "customer_name")],
            "order_by": [_order_output("customer_name", "asc")],
        },
    )

    assert result["referenced_columns"] == [
        {
            "usage": "order_by",
            "table": "customer",
            "column": "first_name",
            "visibility": "user_visible",
            "is_handle": False,
            "is_primary_key": False,
            "table_primary_key": ["customer_id"],
            "direction": "asc",
            "output": "customer_name",
        }
    ]


@pytest.mark.asyncio
async def test_query_aggregate_on_joined_table_uses_explicit_group_ref():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": _from("rental", "r"),
            "join": [
                {"via_edge": "rental.inventory_id->inventory", "as": "i"},
                {"via_edge": "inventory.film_id->film", "as": "f"},
            ],
            "group_by": [_group("f", "film_id")],
            "aggregate": [_agg("count", "rentals")],
            "order_by": [_order_output("rentals", "desc")],
            "limit": 5,
        },
    )
    sql, _ = conn.calls[0]
    assert "GROUP BY t2.\"film_id\"" in sql
    assert "COUNT(*) AS \"rentals\"" in sql


# ---------- validation ----------


@pytest.mark.asyncio
async def test_query_rejects_missing_from_table():
    session, _ = _stub_session()
    with pytest.raises(TypeError):
        await query(session, spec={})


@pytest.mark.asyncio
async def test_query_rejects_legacy_string_from():
    session, _ = _stub_session()
    with pytest.raises(TypeError):
        await query(session, spec={"from": "customer"})


@pytest.mark.asyncio
async def test_query_rejects_legacy_filter_key():
    session, _ = _stub_session()
    with pytest.raises(TypeError):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "filter": [{"column": "customer_id", "op": "eq", "value": 1}],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_unknown_table():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await query(session, spec={"from": _from("nope", "n")})


@pytest.mark.asyncio
async def test_query_rejects_blocked_non_handle_column_refs():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "select": [_select("c", "api_token")],
            },
        )
    with pytest.raises(KeyError):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "where": [
                    {"ref": _ref("c", "api_token"), "op": "eq", "value": "secret"},
                ],
                "aggregate": [_agg("count", "matches")],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_duplicate_join_alias():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": _from("customer", "x"),
                "join": [{"via_edge": "customer<-rental.customer_id", "as": "x"}],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_select_and_aggregate_together():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": _from("rental", "r"),
                "select": [_select("r", "rental_id")],
                "aggregate": [_agg("count", "n")],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_group_by_without_aggregate():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": _from("rental", "r"),
                "group_by": [_group("r", "customer_id")],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_unknown_alias():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "select": [_select("missing", "customer_id")],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_unknown_select_column():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "select": [_select("c", "not_a_column")],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_join_edge_not_originating_at_current_table():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "join": [{"via_edge": "rental.inventory_id->inventory", "as": "i"}],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_unsupported_aggregate_fn():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": _from("rental", "r"),
                "aggregate": [
                    _agg("median", "m", alias="r", column="rental_id"),
                ],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_duplicate_aggregate_outputs():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": _from("rental", "r"),
                "aggregate": [
                    _agg("count", "n"),
                    _agg("max", "n", alias="r", column="rental_date"),
                ],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_non_positive_limit():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(session, spec={"from": _from("customer", "c"), "limit": 0})


@pytest.mark.asyncio
async def test_query_rejects_invalid_order_direction():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "order_by": [_order_ref("c", "customer_id", "up")],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_more_than_two_order_keys():
    session, _ = _stub_session()
    with pytest.raises(ValueError, match="at most two order objects"):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "order_by": [
                    _order_ref("c", "first_name", "asc"),
                    _order_ref("c", "last_name", "asc"),
                    _order_ref("c", "customer_id", "asc"),
                ],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_order_by_with_both_ref_and_output():
    session, _ = _stub_session()
    with pytest.raises(TypeError):
        await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "order_by": [
                    {
                        "ref": _ref("c", "customer_id"),
                        "output": "customer_id",
                        "direction": "asc",
                    }
                ],
            },
        )


# ---------- integration against pagila ----------


async def _live_session() -> tuple[ComposerSession, asyncpg.Connection]:
    config = load_config(Path("rl_task_foundry.yaml"))
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.visibility.default_visibility,
        visibility_overrides=config.visibility.visibility_overrides,
    )
    graph = await introspector.introspect()
    snap = snapshot_from_graph(graph)
    conn = await asyncpg.connect(config.database.dsn)
    await _apply_session_settings(conn, solver_session_settings(config.database))
    return ComposerSession(snapshot=snap, connection=conn), conn


@pytest.mark.asyncio
async def test_query_top_films_by_rental_count_against_pagila():
    session, conn = await _live_session()
    try:
        result = await query(
            session,
            spec={
                "from": _from("rental", "r"),
                "join": [
                    {"via_edge": "rental.inventory_id->inventory", "as": "i"},
                    {"via_edge": "inventory.film_id->film", "as": "f"},
                ],
                "group_by": [_group("f", "title")],
                "aggregate": [_agg("count", "rentals")],
                "order_by": [_order_output("rentals", "desc")],
                "limit": 3,
            },
        )
        rows = result["rows"]
        assert isinstance(rows, list)
        assert len(rows) == 3
        counts = [row["rentals"] for row in rows if isinstance(row, dict)]
        assert counts == sorted(counts, reverse=True)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_query_filter_and_select_against_pagila():
    session, conn = await _live_session()
    try:
        result = await query(
            session,
            spec={
                "from": _from("customer", "c"),
                "where": [
                    {"ref": _ref("c", "store_id"), "op": "eq", "value": "1"},
                ],
                "select": [
                    _select("c", "customer_id"),
                    _select("c", "first_name"),
                ],
                "order_by": [_order_ref("c", "customer_id", "asc")],
                "limit": 5,
            },
        )
        rows = result["rows"]
        assert isinstance(rows, list)
        assert len(rows) == 5
        ids = [row["customer_id"] for row in rows if isinstance(row, dict)]
        assert ids == sorted(cast(list[int], ids))
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_query_aggregate_max_rental_date_against_pagila():
    session, conn = await _live_session()
    try:
        result = await query(
            session,
            spec={
                "from": _from("rental", "r"),
                "aggregate": [
                    _agg("count", "total"),
                    _agg("max", "last", alias="r", column="rental_date"),
                ],
            },
        )
        rows = result["rows"]
        assert isinstance(rows, list)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, dict)
        assert isinstance(row["total"], int)
        assert row["total"] > 0
    finally:
        await conn.close()
