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
        readonly_sql=(),
        timeout_sql=(
            f"SET statement_timeout = {config.statement_timeout_ms}",
            f"SET lock_timeout = {config.lock_timeout_ms}",
            f"SET idle_in_transaction_session_timeout = {config.idle_tx_timeout_ms}",
            "SET search_path = public",
        ),
    )


def solver_session_settings(config: DatabaseConfig) -> SessionSettings:
    """Return SQL statements required for a safe solver session."""

    readonly_sql = ["SET default_transaction_read_only = on"]
    if config.readonly_role:
        readonly_sql.append(f"SET ROLE {config.readonly_role}")
    base = control_session_settings(config)
    return SessionSettings(readonly_sql=tuple(readonly_sql), timeout_sql=base.timeout_sql)


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


async def smoke_test_connection(config: DatabaseConfig) -> dict[str, str]:
    """Open a single connection and return a tiny identity payload."""

    conn = await asyncpg.connect(dsn=config.dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT
                current_database()::text AS database_name,
                current_user::text AS user_name,
                current_schema()::text AS schema_name
            """
        )
        assert row is not None
        return {
            "database_name": row["database_name"],
            "user_name": row["user_name"],
            "schema_name": row["schema_name"],
        }
    finally:
        await conn.close()
