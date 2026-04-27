"""Integration tests for the atomic calculus against the live pagila DB.

Each test builds a fresh AtomicSession, exercises a primitive or chain,
and tears the connection down in a finally block. Tests are skipped
implicitly by `PostgresSchemaIntrospector` / `asyncpg.connect` raising
when the DB is unreachable; no explicit skip logic needed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import asyncpg
import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.infra.db import (
    _apply_session_settings,
    solver_session_settings,
)
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.tooling.atomic import AtomicSession, CursorStore
from rl_task_foundry.tooling.atomic.calculus import (
    aggregate,
    count,
    create_row_set,
    filter_rows,
    group_top,
    intersect,
    read,
    rows_via,
    rows_where,
    take,
)
from rl_task_foundry.tooling.atomic.cursor import order_by
from rl_task_foundry.tooling.common import snapshot_from_graph


async def _build_session():
    config = load_config(Path("rl_task_foundry.yaml"))
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.visibility.default_visibility,
        visibility_overrides=config.visibility.visibility_overrides,
    )
    graph = await introspector.introspect()
    snapshot = snapshot_from_graph(graph)

    conn = await asyncpg.connect(config.database.dsn)
    await _apply_session_settings(conn, solver_session_settings(config.database))
    session = AtomicSession(
        snapshot=snapshot,
        connection=conn,
        store=CursorStore(),
    )
    return session, conn


@pytest.mark.asyncio
async def test_vertical_slice_over_pagila():
    session, conn = await _build_session()
    try:
        rentals_of_customer_45 = rows_where(
            session,
            table="rental",
            column="customer_id",
            op="eq",
            value=45,
        )

        ordered = order_by(
            session.store,
            rentals_of_customer_45,
            column="rental_date",
            direction="asc",
        )

        ids = await take(session, cursor=ordered, n=3)
        assert len(ids) == 3
        assert all(isinstance(rid, int) for rid in ids)

        first = await read(
            session,
            table="rental",
            row_id=ids[0],
            columns=["rental_id", "customer_id", "rental_date"],
        )
        assert first["rental_id"] == ids[0]
        assert first["customer_id"] == 45
        assert "rental_date" in first
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_take_allows_limit_one_and_rejects_invalid_api_limits():
    session, conn = await _build_session()
    try:
        cursor = rows_where(
            session, table="rental", column="customer_id", op="eq", value=45
        )
        one = await take(session, cursor=cursor, n=1)
        assert len(one) == 1
        with pytest.raises(ValueError):
            await take(session, cursor=cursor, n=0)
        with pytest.raises(ValueError):
            await take(session, cursor=cursor, n=session.max_fetch_limit + 1)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_rows_where_rejects_unknown_column():
    session, conn = await _build_session()
    try:
        with pytest.raises(KeyError):
            rows_where(
                session,
                table="rental",
                column="nonexistent",
                op="eq",
                value=1,
            )
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_rows_via_forward_projects_rentals_to_customer():
    session, conn = await _build_session()
    try:
        # Customer 45's rentals → their customer row (all map back to 45).
        cursor = rows_where(
            session,
            table="rental",
            column="customer_id",
            op="eq",
            value=45,
        )
        via = rows_via(
            session,
            cursor=cursor,
            edge_label="rental.customer_id->customer",
        )
        ids = await take(session, cursor=via, n=2)
        # Dedup in take → a single customer row.
        assert set(ids) == {45}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_rows_via_reverse_projects_customer_to_rentals():
    session, conn = await _build_session()
    try:
        customers = rows_where(
            session,
            table="customer",
            column="customer_id",
            op="eq",
            value=45,
        )
        rentals = rows_via(
            session,
            cursor=customers,
            edge_label="customer<-rental.customer_id",
        )
        cnt = await count(session, cursor=rentals)
        assert cnt > 0
        ordered = order_by(
            session.store, rentals, column="rental_date", direction="desc"
        )
        ids = await take(session, cursor=ordered, n=3)
        assert len(ids) == 3
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_relation_round_trip_keeps_record_set_semantics():
    session, conn = await _build_session()
    try:
        rentals = create_row_set(session, table="rental")
        inventories_from_rentals = rows_via(
            session,
            cursor=rentals,
            edge_label="rental.inventory_id->inventory",
        )
        matching_inventory = filter_rows(
            session,
            cursor=inventories_from_rentals,
            column="film_id",
            op="eq",
            value=308,
        )
        matching_rentals = rows_via(
            session,
            cursor=matching_inventory,
            edge_label="inventory<-rental.inventory_id",
        )
        h2_matching_rentals = filter_rows(
            session,
            cursor=matching_rentals,
            column="rental_date",
            op="gte",
            value=datetime(2022, 7, 1),
        )
        bounded_h2_matching_rentals = filter_rows(
            session,
            cursor=h2_matching_rentals,
            column="rental_date",
            op="lt",
            value=datetime(2023, 1, 1),
        )

        assert await count(session, cursor=h2_matching_rentals) == 12
        assert await count(session, cursor=bounded_h2_matching_rentals) == 12
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_timestamptz_date_strings_match_utc_canonical_query():
    session, conn = await _build_session()
    try:
        rentals = create_row_set(session, table="rental")
        july_or_later = filter_rows(
            session,
            cursor=rentals,
            column="rental_date",
            op="gte",
            value="2022-07-01",
        )
        july = filter_rows(
            session,
            cursor=july_or_later,
            column="rental_date",
            op="lt",
            value="2022-08-01",
        )
        payments = rows_via(
            session,
            cursor=july,
            edge_label="rental<-payment.rental_id",
        )

        total = await aggregate(session, cursor=payments, fn="sum", column="amount")

        assert str(total) == "28510.56"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_intersect_rejects_mismatched_targets():
    session, conn = await _build_session()
    try:
        customers = rows_where(
            session, table="customer", column="store_id", op="eq", value=1
        )
        rentals = rows_where(
            session,
            table="rental",
            column="customer_id",
            op="eq",
            value=45,
        )
        with pytest.raises(ValueError):
            intersect(session, left=customers, right=rentals)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_intersect_over_customer_filters():
    session, conn = await _build_session()
    try:
        store_1 = rows_where(
            session, table="customer", column="store_id", op="eq", value=1
        )
        active = rows_where(
            session, table="customer", column="active", op="eq", value=1
        )
        combined = intersect(session, left=store_1, right=active)
        only_store_1 = await count(session, cursor=store_1)
        overlap = await count(session, cursor=combined)
        assert 0 < overlap <= only_store_1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_aggregate_max_rental_date_of_customer_45():
    session, conn = await _build_session()
    try:
        cursor = rows_where(
            session,
            table="rental",
            column="customer_id",
            op="eq",
            value=45,
        )
        last = await aggregate(
            session, cursor=cursor, fn="max", column="rental_date"
        )
        assert last is not None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_group_top_count_returns_top_customers_by_rentals():
    session, conn = await _build_session()
    try:
        cursor = rows_where(
            session,
            table="rental",
            column="rental_date",
            op="gt",
            value=datetime(2005, 1, 1),
        )
        tops = await group_top(
            session,
            cursor=cursor,
            group_column="customer_id",
            fn="count",
            n=3,
        )
        assert len(tops) == 3
        counts: list[int] = []
        for _, value in tops:
            assert isinstance(value, int)
            counts.append(value)
        assert counts == sorted(counts, reverse=True)
    finally:
        await conn.close()
