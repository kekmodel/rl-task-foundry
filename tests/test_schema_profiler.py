from __future__ import annotations

import pytest

from rl_task_foundry.config.models import DatabaseConfig
from rl_task_foundry.schema import profiler
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph, TableProfile
from rl_task_foundry.schema.profiler import DataProfile, _profile_table, profile_database


def _column(
    table: str,
    name: str,
    data_type: str,
    *,
    schema: str,
    is_primary_key: bool = False,
    is_foreign_key: bool = False,
    n_distinct: float | None = None,
) -> ColumnProfile:
    return ColumnProfile(
        schema_name=schema,
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=1,
        is_nullable=False,
        visibility="user_visible",
        is_primary_key=is_primary_key,
        is_foreign_key=is_foreign_key,
        n_distinct=n_distinct,
    )


class _FakeConnection:
    def __init__(self) -> None:
        self.fetchrow_sql: list[str] = []
        self.fetch_sql: list[str] = []
        self.closed = False

    async def fetchrow(self, sql: str, *params: object) -> dict[str, object]:
        self.fetchrow_sql.append(sql)
        assert params == ()
        return {
            "mean": 10.0,
            "std": 2.0,
            "min_val": 1.0,
            "max_val": 20.0,
            "distinct_count": 5,
        }

    async def fetch(self, sql: str, *params: object) -> list[dict[str, object]]:
        self.fetch_sql.append(sql)
        assert params == ()
        return [
            {"val": "open", "cnt": 3},
            {"val": "closed", "cnt": 2},
        ]

    async def close(self) -> None:
        self.closed = True


async def test_profile_table_quotes_schema_table_and_column_identifiers() -> None:
    table = TableProfile(
        schema_name='tenant "A"',
        table_name="order detail",
        columns=[
            _column(
                "order detail",
                "id",
                "integer",
                schema='tenant "A"',
                is_primary_key=True,
            ),
            _column(
                "order detail",
                "gross$total",
                "integer",
                schema='tenant "A"',
            ),
            _column(
                "order detail",
                "state.name",
                "text",
                schema='tenant "A"',
                n_distinct=2,
            ),
        ],
        primary_key=("id",),
        row_estimate=25,
    )
    conn = _FakeConnection()
    profile = DataProfile()

    await _profile_table(conn, table, profile)  # type: ignore[arg-type]

    assert conn.fetchrow_sql == [
        'SELECT avg("gross$total"::float) AS mean, '
        'stddev("gross$total"::float) AS std, '
        'min("gross$total"::float) AS min_val, '
        'max("gross$total"::float) AS max_val, '
        'count(distinct "gross$total") AS distinct_count '
        'FROM "tenant ""A"""."order detail" '
        'WHERE "gross$total" IS NOT NULL'
    ]
    assert conn.fetch_sql == [
        'SELECT "state.name" AS val, count(*) AS cnt '
        'FROM "tenant ""A"""."order detail" '
        'WHERE "state.name" IS NOT NULL '
        'GROUP BY "state.name" ORDER BY cnt DESC LIMIT 20'
    ]
    assert profile.numeric[0].column == "gross$total"
    assert profile.categorical[0].categories == ["open", "closed"]


async def test_profile_database_profiles_unknown_row_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TableProfile(
        schema_name="public",
        table_name="payment",
        columns=[
            _column("payment", "payment_id", "integer", schema="public", is_primary_key=True),
            _column("payment", "amount", "numeric", schema="public"),
        ],
        primary_key=("payment_id",),
        row_estimate=None,
    )
    graph = SchemaGraph(tables=[table])
    conn = _FakeConnection()

    async def fake_connect(dsn: str) -> _FakeConnection:
        assert dsn == "postgresql://example"
        return conn

    async def fake_apply_session_settings(
        connection: _FakeConnection,
        settings: object,
    ) -> None:
        assert connection is conn
        assert settings is not None

    monkeypatch.setattr(profiler.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(profiler, "_apply_session_settings", fake_apply_session_settings)

    profile = await profile_database(
        DatabaseConfig(
            dsn="postgresql://example",
            readonly_role="readonly",
        ),
        graph,
    )

    assert profile.numeric[0].table == "public.payment"
    assert profile.numeric[0].column == "amount"
    assert conn.closed is True


async def test_profile_database_skips_tables_with_postgres_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TableProfile(
        schema_name="proof_trial",
        table_name="customer",
        columns=[
            _column("customer", "name", "text", schema="proof_trial", n_distinct=2),
        ],
        row_estimate=None,
    )
    graph = SchemaGraph(tables=[table])
    conn = _FakeConnection()

    async def fake_connect(dsn: str) -> _FakeConnection:
        assert dsn == "postgresql://example"
        return conn

    async def fake_apply_session_settings(
        connection: _FakeConnection,
        settings: object,
    ) -> None:
        assert connection is conn

    async def fake_profile_table(
        connection: _FakeConnection,
        table: TableProfile,
        profile: DataProfile,
    ) -> None:
        raise profiler.asyncpg.PostgresError("permission denied")

    monkeypatch.setattr(profiler.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(profiler, "_apply_session_settings", fake_apply_session_settings)
    monkeypatch.setattr(profiler, "_profile_table", fake_profile_table)

    profile = await profile_database(
        DatabaseConfig(
            dsn="postgresql://example",
            readonly_role="readonly",
        ),
        graph,
    )

    assert profile.numeric == []
    assert profile.categorical == []
    assert conn.closed is True
