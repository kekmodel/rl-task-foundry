"""Data distribution profiler for constraint-rich task synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field

import asyncpg

from rl_task_foundry.config.models import DatabaseConfig
from rl_task_foundry.infra.db import _apply_session_settings, control_session_settings
from rl_task_foundry.schema.graph import SchemaGraph, TableProfile
from rl_task_foundry.tooling.common.sql import quote_ident, quote_table, readonly_select


@dataclass(frozen=True, slots=True)
class NumericColumnStats:
    table: str
    column: str
    mean: float
    std: float
    min_val: float
    max_val: float
    distinct: int
    low_threshold: float
    high_threshold: float

    def render(self) -> str:
        return (
            f"- {self.table}.{self.column}: "
            f"low(<{self.low_threshold:.2f}) "
            f"mid({self.low_threshold:.2f}-{self.high_threshold:.2f}) "
            f"high(>{self.high_threshold:.2f})"
        )


@dataclass(frozen=True, slots=True)
class CategoricalColumnStats:
    table: str
    column: str
    categories: list[str]
    counts: list[int]

    def render(self) -> str:
        items = ", ".join(
            f"{cat}({cnt})" for cat, cnt in zip(self.categories, self.counts)
        )
        return f"- {self.table}.{self.column}: {len(self.categories)} categories [{items}]"


@dataclass(slots=True)
class DataProfile:
    numeric: list[NumericColumnStats] = field(default_factory=list)
    categorical: list[CategoricalColumnStats] = field(default_factory=list)

    def render(self) -> str:
        lines = []
        for stat in self.numeric:
            lines.append(stat.render())
        for stat in self.categorical:
            lines.append(stat.render())
        return "\n".join(lines)


_NUMERIC_TYPES = frozenset({
    "integer", "bigint", "smallint",
    "numeric", "decimal", "real", "double precision",
    "money",
})

_CATEGORICAL_MAX_DISTINCT = 20


async def profile_database(
    database: DatabaseConfig,
    graph: SchemaGraph,
) -> DataProfile:
    conn = await asyncpg.connect(database.dsn)
    try:
        await _apply_session_settings(conn, control_session_settings(database))
        profile = DataProfile()

        for table in graph.tables:
            if table.row_estimate is not None and table.row_estimate < 10:
                continue
            try:
                await _profile_table(conn, table, profile)
            except asyncpg.PostgresError:
                continue

        return profile
    finally:
        await conn.close()


async def _profile_table(
    conn: asyncpg.Connection,
    table: TableProfile,
    profile: DataProfile,
) -> None:
    qualified = f"{table.schema_name}.{table.table_name}"
    quoted_table = quote_table(table.schema_name, table.table_name)

    for col in table.columns:
        quoted_column = quote_ident(col.column_name)
        if col.is_primary_key or col.is_foreign_key:
            continue

        if col.data_type in _NUMERIC_TYPES:
            row = await conn.fetchrow(
                readonly_select(
                    f"SELECT avg({quoted_column}::float) AS mean, "
                    f"stddev({quoted_column}::float) AS std, "
                    f"min({quoted_column}::float) AS min_val, "
                    f"max({quoted_column}::float) AS max_val, "
                    f"count(distinct {quoted_column}) AS distinct_count "
                    f"FROM {quoted_table} "
                    f"WHERE {quoted_column} IS NOT NULL"
                )
            )
            if row and row["mean"] is not None and row["std"] is not None:
                mean = float(row["mean"])
                std = float(row["std"])
                if std > 0:
                    profile.numeric.append(NumericColumnStats(
                        table=qualified,
                        column=col.column_name,
                        mean=mean,
                        std=std,
                        min_val=float(row["min_val"]),
                        max_val=float(row["max_val"]),
                        distinct=int(row["distinct_count"]),
                        low_threshold=round(mean - std, 2),
                        high_threshold=round(mean + std, 2),
                    ))

        else:
            if col.data_type in _NUMERIC_TYPES:
                continue
            distinct_count = col.n_distinct
            if distinct_count is not None and distinct_count < 0:
                distinct_count = abs(distinct_count) * (table.row_estimate or 0)
            if distinct_count is not None and distinct_count > _CATEGORICAL_MAX_DISTINCT:
                continue
            if distinct_count is not None and distinct_count < 2:
                continue
            rows = await conn.fetch(
                readonly_select(
                    f"SELECT {quoted_column} AS val, count(*) AS cnt "
                    f"FROM {quoted_table} "
                    f"WHERE {quoted_column} IS NOT NULL "
                    f"GROUP BY {quoted_column} "
                    f"ORDER BY cnt DESC "
                    f"LIMIT {_CATEGORICAL_MAX_DISTINCT}"
                )
            )
            if len(rows) >= 2:
                categories = [str(r["val"]) for r in rows]
                counts = [int(r["cnt"]) for r in rows]
                profile.categorical.append(CategoricalColumnStats(
                    table=qualified,
                    column=col.column_name,
                    categories=categories,
                    counts=counts,
                ))
