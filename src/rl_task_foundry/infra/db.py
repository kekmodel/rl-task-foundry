"""Database lane configuration helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import asyncpg

from rl_task_foundry.config.models import DatabaseConfig


@dataclass(slots=True)
class SessionSettings:
    readonly_sql: tuple[str, ...]
    timeout_sql: tuple[str, ...]


def control_session_settings(config: DatabaseConfig) -> SessionSettings:
    """Return SQL statements required for a safe control-plane session."""

    return SessionSettings(
        readonly_sql=_readonly_sql(config),
        timeout_sql=(
            f"SET statement_timeout = {config.statement_timeout_ms}",
            f"SET lock_timeout = {config.lock_timeout_ms}",
            f"SET idle_in_transaction_session_timeout = {config.idle_tx_timeout_ms}",
            "SET search_path = public",
        ),
    )


def mutating_control_session_settings(config: DatabaseConfig) -> SessionSettings:
    """Return settings for explicit fixture/setup lanes that must write.

    Normal control/composer sessions are read-only. This helper exists so
    test fixture DDL and other explicit maintenance operations do not weaken
    the default safety contract by reusing the read-only control lane.
    """

    base = control_session_settings(config)
    return SessionSettings(readonly_sql=(), timeout_sql=base.timeout_sql)


def solver_session_settings(config: DatabaseConfig) -> SessionSettings:
    """Return SQL statements required for a safe solver session."""

    readonly_sql = list(_readonly_sql(config))
    base = control_session_settings(config)
    return SessionSettings(readonly_sql=tuple(readonly_sql), timeout_sql=base.timeout_sql)


def _readonly_sql(config: DatabaseConfig) -> tuple[str, ...]:
    readonly_sql = ["SET default_transaction_read_only = on"]
    if config.readonly_role:
        readonly_sql.append(f"SET ROLE {config.readonly_role}")
    return tuple(readonly_sql)


async def _apply_session_settings(conn: Any, settings: SessionSettings) -> None:
    for statement in (*settings.readonly_sql, *settings.timeout_sql):
        try:
            await conn.execute(statement)
        except asyncpg.PostgresError as exc:
            if _should_ignore_session_setting_error(statement, exc):
                continue
            raise


def _should_ignore_session_setting_error(
    statement: str,
    exc: asyncpg.PostgresError,
) -> bool:
    if not statement.startswith("SET ROLE "):
        return False
    if isinstance(exc, (asyncpg.InvalidParameterValueError, asyncpg.UndefinedObjectError)):
        return True
    detail = str(exc).lower()
    return "role" in detail and ("does not exist" in detail or "invalid" in detail)


@dataclass(slots=True)
class DatabasePools:
    """Two-lane pool with solver and control connections."""

    solver_pool: Any
    control_pool: Any

    @classmethod
    async def create(cls, config: DatabaseConfig) -> "DatabasePools":
        solver_settings = solver_session_settings(config)
        control_settings = control_session_settings(config)
        solver_pool = await asyncpg.create_pool(
            dsn=config.dsn,
            min_size=1,
            max_size=config.solver_pool_size,
            init=lambda conn: _apply_session_settings(conn, solver_settings),
        )
        control_pool = await asyncpg.create_pool(
            dsn=config.dsn,
            min_size=1,
            max_size=config.control_pool_size,
            init=lambda conn: _apply_session_settings(conn, control_settings),
        )
        return cls(
            solver_pool=solver_pool,
            control_pool=control_pool,
        )

    @asynccontextmanager
    async def solver_connection(self):
        async with self.solver_pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def control_connection(self):
        async with self.control_pool.acquire() as conn:
            yield conn

    async def close(self) -> None:
        await self.solver_pool.close()
        await self.control_pool.close()


async def ensure_database_pools(
    existing: DatabasePools | None,
    config: DatabaseConfig,
) -> DatabasePools:
    if existing is not None:
        return existing
    return await DatabasePools.create(config)


async def ensure_attached_database_pools(
    owner: Any,
    *,
    attr_name: str,
    config: DatabaseConfig,
) -> DatabasePools:
    existing = getattr(owner, attr_name)
    pools = await ensure_database_pools(existing, config)
    if pools is not existing:
        setattr(owner, attr_name, pools)
    return pools


async def smoke_test_connection(config: DatabaseConfig) -> dict[str, str]:
    """Open a connection and verify the configured read-only schema surface."""

    conn = await asyncpg.connect(dsn=config.dsn)
    try:
        await _apply_session_settings(conn, control_session_settings(config))
        row = await conn.fetchrow(
            """
            SELECT
                current_database()::text AS database_name,
                current_user::text AS user_name,
                current_schema()::text AS schema_name,
                current_setting('default_transaction_read_only')::text AS read_only
            """
        )
        assert row is not None
        schema_rows = await conn.fetch(
            """
            WITH wanted AS (
                SELECT unnest($1::text[]) AS schema_name
            )
            SELECT
                wanted.schema_name AS requested_schema,
                ns.nspname AS schema_name,
                CASE
                    WHEN ns.oid IS NULL THEN false
                    ELSE has_schema_privilege(current_user, ns.oid, 'USAGE')
                END AS has_usage,
                count(cls.oid) FILTER (
                    WHERE cls.relkind IN ('r', 'p')
                )::int AS table_count,
                count(cls.oid) FILTER (
                    WHERE cls.relkind IN ('r', 'p')
                      AND has_table_privilege(current_user, cls.oid, 'SELECT')
                )::int AS selectable_table_count
            FROM wanted
            LEFT JOIN pg_namespace AS ns
              ON ns.nspname = wanted.schema_name
            LEFT JOIN pg_class AS cls
              ON cls.relnamespace = ns.oid
             AND cls.relkind IN ('r', 'p')
            GROUP BY wanted.schema_name, ns.nspname, ns.oid
            ORDER BY wanted.schema_name
            """,
            config.schema_allowlist,
        )
        issues: list[str] = []
        total_tables = 0
        total_selectable_tables = 0
        for schema_row in schema_rows:
            requested_schema = schema_row["requested_schema"]
            schema_name = schema_row["schema_name"]
            has_usage = bool(schema_row["has_usage"])
            table_count = int(schema_row["table_count"])
            selectable_table_count = int(schema_row["selectable_table_count"])
            total_tables += table_count
            total_selectable_tables += selectable_table_count
            if schema_name is None:
                issues.append(f"schema {requested_schema!r} is missing")
                continue
            if not has_usage:
                issues.append(f"schema {requested_schema!r} lacks USAGE for current_user")
            if table_count != selectable_table_count:
                issues.append(
                    f"schema {requested_schema!r} has {table_count} tables but "
                    f"only {selectable_table_count} are SELECT-able"
                )
        if issues:
            raise RuntimeError("database allowlist smoke failed: " + "; ".join(issues))
        return {
            "database_name": row["database_name"],
            "user_name": row["user_name"],
            "schema_name": row["schema_name"],
            "read_only": row["read_only"],
            "schema_allowlist": ",".join(config.schema_allowlist),
            "allowlisted_table_count": str(total_tables),
            "selectable_table_count": str(total_selectable_tables),
        }
    finally:
        await conn.close()
