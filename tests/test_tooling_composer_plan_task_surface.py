"""Tests for tooling.composer.plan_task_surface."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import pytest

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.tooling.common import snapshot_from_graph
from rl_task_foundry.tooling.composer import ComposerSession, plan_task_surface
from rl_task_foundry.tooling.composer._session import _Row


class _ScriptedConnection:
    def __init__(self) -> None:
        self.fetch_results: list[list[dict[str, object]]] = []
        self.fetchrow_results: list[dict[str, object] | None] = []
        self.fetchval_results: list[object] = []
        self.calls: list[tuple[str, str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> Sequence[_Row]:
        self.calls.append(("fetch", sql, args))
        payload = self.fetch_results.pop(0)
        return cast(Sequence[_Row], list(payload))

    async def fetchrow(self, sql: str, *args: object):
        self.calls.append(("fetchrow", sql, args))
        payload = self.fetchrow_results.pop(0)
        if payload is None:
            return None
        return cast(_Row, payload)

    async def fetchval(self, sql: str, *args: object):
        self.calls.append(("fetchval", sql, args))
        return self.fetchval_results.pop(0)


def _column(
    table: str,
    name: str,
    data_type: str = "integer",
    *,
    is_primary_key: bool = False,
    is_foreign_key: bool = False,
    is_nullable: bool = False,
    visibility: str = "user_visible",
) -> ColumnProfile:
    return ColumnProfile(
        schema_name="public",
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=1,
        is_nullable=is_nullable,
        visibility=visibility,  # type: ignore[arg-type]
        is_primary_key=is_primary_key,
        is_foreign_key=is_foreign_key,
    )


def _snapshot(*, no_pk_child: bool = False):
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "customer_id", is_primary_key=True),
            _column("customer", "name", data_type="text"),
            _column("customer", "secret", data_type="text", visibility="blocked"),
        ],
        primary_key=("customer_id",),
    )
    child_primary_key = () if no_pk_child else ("rental_id",)
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            _column("rental", "rental_id", is_primary_key=not no_pk_child),
            _column("rental", "customer_id", is_foreign_key=True),
            _column("rental", "rental_date", data_type="timestamp"),
            _column("rental", "amount", data_type="numeric", is_nullable=True),
            _column("rental", "notes", data_type="text", is_nullable=True),
            _column("rental", "internal_hash", data_type="text", visibility="blocked"),
        ],
        primary_key=child_primary_key,
    )
    edge = ForeignKeyEdge(
        constraint_name="rental_customer",
        source_schema="public",
        source_table="rental",
        source_columns=("customer_id",),
        target_schema="public",
        target_table="customer",
        target_columns=("customer_id",),
    )
    return snapshot_from_graph(
        SchemaGraph(tables=[customer, rental], edges=[edge])
    )


def _session(*, no_pk_child: bool = False) -> tuple[ComposerSession, _ScriptedConnection]:
    conn = _ScriptedConnection()
    session = ComposerSession(
        snapshot=_snapshot(no_pk_child=no_pk_child),
        connection=conn,
    )
    return session, conn


def _event_surface_snapshot():
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "customer_id", is_primary_key=True),
            _column("customer", "name", data_type="text"),
        ],
        primary_key=("customer_id",),
    )
    event = TableProfile(
        schema_name="public",
        table_name="event",
        columns=[
            _column("event", "event_id", is_primary_key=True),
            _column("event", "customer_id", is_foreign_key=True),
            _column("event", "event_time", data_type="timestamp"),
        ],
        primary_key=("event_id",),
    )
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            _column("rental", "rental_id", is_primary_key=True),
            _column("rental", "customer_id", is_foreign_key=True),
            _column("rental", "rental_date", data_type="timestamp"),
            _column("rental", "amount", data_type="numeric"),
        ],
        primary_key=("rental_id",),
    )
    edges = [
        ForeignKeyEdge(
            constraint_name="event_customer",
            source_schema="public",
            source_table="event",
            source_columns=("customer_id",),
            target_schema="public",
            target_table="customer",
            target_columns=("customer_id",),
        ),
        ForeignKeyEdge(
            constraint_name="rental_customer",
            source_schema="public",
            source_table="rental",
            source_columns=("customer_id",),
            target_schema="public",
            target_table="customer",
            target_columns=("customer_id",),
        ),
    ]
    return snapshot_from_graph(
        SchemaGraph(tables=[customer, event, rental], edges=edges)
    )


def _event_surface_session() -> tuple[ComposerSession, _ScriptedConnection]:
    conn = _ScriptedConnection()
    session = ComposerSession(
        snapshot=_event_surface_snapshot(),
        connection=conn,
    )
    return session, conn


@pytest.mark.asyncio
async def test_plan_task_surface_returns_structural_reverse_candidate():
    session, conn = _session()
    conn.fetchrow_results = [{"customer_id": 45, "name": "ALICE"}]
    conn.fetchval_results = [4]
    conn.fetch_results = [
        [
            {"rental_date": "2020-01-01T00:00:00", "amount": None, "notes": "alpha"},
            {
                "rental_date": "2020-01-02T00:00:00",
                "amount": "12.50",
                "notes": None,
            },
        ]
    ]

    payload = await plan_task_surface(
        session,
        table="customer",
        row_id=45,
        max_sample_rows=2,
    )

    assert payload["anchor"] == {"table": "customer", "row_id": 45}
    assert payload["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["surface_path"] == [
        {
            "from_table": "customer",
            "via_edge": "customer<-rental.customer_id",
            "direction": "reverse",
            "to_table": "rental",
        }
    ]
    assert candidate["record_surface"] == {
        "table": "rental",
        "primary_key_backed": True,
        "record_revisitable": True,
        "relationship_direction": "reverse",
        "path_kind": "direct",
        "match_columns": ["customer_id"],
        "total_count": 4,
        "sample_size": 2,
    }
    outputs = {
        output["field"]: output for output in candidate["candidate_outputs"]
    }
    assert set(outputs) == {"amount", "notes", "rental_date"}
    assert outputs["amount"]["sample_non_null"] == 1
    assert outputs["notes"]["sample_non_null"] == 1
    orders = {
        order["field"]: order for order in candidate["candidate_orders"]
    }
    assert set(orders) == {"amount", "rental_date"}
    assert orders["rental_date"]["directions"] == ["asc", "desc"]
    assert orders["rental_date"]["requires_user_phrase"] is True
    assert "order_ties_must_be_checked_by_final_query" in candidate[
        "structural_risks"
    ]
    assert payload["notes"] == [
        "planning_only",
        "not_label_evidence",
        "final_query_required",
    ]
    serialized = json.dumps(payload, sort_keys=True)
    assert "ALICE" not in serialized
    assert "alpha" not in serialized
    assert "12.50" not in serialized


@pytest.mark.asyncio
async def test_plan_task_surface_marks_no_pk_child_as_not_revisitable():
    session, conn = _session(no_pk_child=True)
    conn.fetchrow_results = [{"customer_id": 45, "name": "ALICE"}]
    conn.fetchval_results = [3]

    payload = await plan_task_surface(
        session,
        table="customer",
        row_id=45,
    )

    candidate = payload["candidates"][0]
    record_surface = candidate["record_surface"]
    assert record_surface["table"] == "rental"
    assert record_surface["primary_key_backed"] is False
    assert record_surface["record_revisitable"] is False
    assert record_surface["sample_size"] == 0
    assert "not_primary_key_backed" in candidate["structural_risks"]
    assert all(call[0] != "fetch" for call in conn.calls)


@pytest.mark.asyncio
async def test_plan_task_surface_reports_zero_rows_for_null_forward_fk():
    session, conn = _session()
    conn.fetchrow_results = [
        {
            "rental_id": 7,
            "customer_id": None,
            "rental_date": "2020-01-01T00:00:00",
            "amount": None,
            "notes": None,
        }
    ]

    payload = await plan_task_surface(
        session,
        table="rental",
        row_id=7,
    )

    candidate = payload["candidates"][0]
    assert candidate["record_surface"]["table"] == "customer"
    assert candidate["record_surface"]["total_count"] == 0
    assert candidate["record_surface"]["sample_size"] == 0
    assert "zero_related_rows" in candidate["structural_risks"]
    assert [call[0] for call in conn.calls] == ["fetchrow"]


@pytest.mark.asyncio
async def test_plan_task_surface_includes_parent_to_sibling_candidates():
    session, conn = _event_surface_session()
    conn.fetchrow_results = [
        {"event_id": 7, "customer_id": 45, "event_time": "2020-01-01T00:00:00"},
        {"customer_id": 45, "name": "ALICE"},
    ]
    conn.fetchval_results = [1, 2, 4]
    conn.fetch_results = [
        [{"name": "ALICE"}],
        [
            {"event_time": "2020-01-01T00:00:00"},
            {"event_time": "2020-01-02T00:00:00"},
        ],
        [
            {"rental_date": "2020-01-01T00:00:00", "amount": "10.00"},
            {"rental_date": "2020-01-02T00:00:00", "amount": "20.00"},
        ],
    ]

    payload = await plan_task_surface(
        session,
        table="event",
        row_id=7,
        max_candidates=5,
        max_sample_rows=2,
    )

    candidates = payload["candidates"]
    sibling = next(
        candidate
        for candidate in candidates
        if candidate["record_surface"]["table"] == "rental"
    )
    assert sibling["record_surface"]["path_kind"] == "via_parent"
    assert sibling["record_surface"]["total_count"] == 4
    assert sibling["surface_path"] == [
        {
            "from_table": "event",
            "via_edge": "event.customer_id->customer",
            "direction": "forward",
            "to_table": "customer",
        },
        {
            "from_table": "customer",
            "via_edge": "customer<-rental.customer_id",
            "direction": "reverse",
            "to_table": "rental",
        },
    ]
    assert {output["field"] for output in sibling["candidate_outputs"]} == {
        "amount",
        "rental_date",
    }
    assert {order["field"] for order in sibling["candidate_orders"]} == {
        "amount",
        "rental_date",
    }
