"""PostgreSQL schema introspection."""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from rl_task_foundry.config.models import DatabaseConfig
from rl_task_foundry.infra.db import control_session_settings
from rl_task_foundry.infra.privacy import Visibility
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.sensitivity import ColumnRef, classify_columns

_TABLE_QUERY = """
SELECT
  ns.nspname AS schema_name,
  cls.relname AS table_name,
  CASE
    WHEN cls.reltuples < 0 THEN NULL
    ELSE cls.reltuples::bigint
  END AS row_estimate
FROM pg_class AS cls
JOIN pg_namespace AS ns
  ON ns.oid = cls.relnamespace
WHERE cls.relkind = 'r'
  AND ns.nspname = ANY($1::text[])
ORDER BY ns.nspname, cls.relname
"""

_COLUMN_QUERY = """
SELECT
  cols.table_schema AS schema_name,
  cols.table_name,
  cols.column_name,
  cols.ordinal_position,
  (cols.is_nullable = 'YES') AS is_nullable,
  COALESCE(NULLIF(cols.udt_name, ''), cols.data_type) AS data_type
FROM information_schema.columns AS cols
JOIN information_schema.tables AS tbl
  ON tbl.table_schema = cols.table_schema
 AND tbl.table_name = cols.table_name
WHERE tbl.table_type = 'BASE TABLE'
  AND cols.table_schema = ANY($1::text[])
ORDER BY cols.table_schema, cols.table_name, cols.ordinal_position
"""

_UNIQUE_INDEX_QUERY = """
SELECT
  ns.nspname AS schema_name,
  tbl.relname AS table_name,
  idx.relname AS index_name,
  pg_idx.indisprimary AS is_primary,
  ARRAY_AGG(att.attname ORDER BY key_cols.ordinality) AS column_names
FROM pg_index AS pg_idx
JOIN pg_class AS tbl
  ON tbl.oid = pg_idx.indrelid
JOIN pg_namespace AS ns
  ON ns.oid = tbl.relnamespace
JOIN pg_class AS idx
  ON idx.oid = pg_idx.indexrelid
JOIN LATERAL unnest(pg_idx.indkey) WITH ORDINALITY AS key_cols(attnum, ordinality)
  ON TRUE
JOIN pg_attribute AS att
  ON att.attrelid = tbl.oid
 AND att.attnum = key_cols.attnum
WHERE pg_idx.indisunique
  AND tbl.relkind = 'r'
  AND ns.nspname = ANY($1::text[])
GROUP BY ns.nspname, tbl.relname, idx.relname, pg_idx.indisprimary
ORDER BY ns.nspname, tbl.relname, idx.relname
"""

_FOREIGN_KEY_QUERY = """
SELECT
  con.conname AS constraint_name,
  src_ns.nspname AS source_schema,
  src_tbl.relname AS source_table,
  tgt_ns.nspname AS target_schema,
  tgt_tbl.relname AS target_table,
  ARRAY_AGG(src_att.attname ORDER BY key_cols.ordinality) AS source_columns,
  ARRAY_AGG(tgt_att.attname ORDER BY key_cols.ordinality) AS target_columns
FROM pg_constraint AS con
JOIN pg_class AS src_tbl
  ON src_tbl.oid = con.conrelid
JOIN pg_namespace AS src_ns
  ON src_ns.oid = src_tbl.relnamespace
JOIN pg_class AS tgt_tbl
  ON tgt_tbl.oid = con.confrelid
JOIN pg_namespace AS tgt_ns
  ON tgt_ns.oid = tgt_tbl.relnamespace
JOIN LATERAL unnest(con.conkey, con.confkey) WITH ORDINALITY AS key_cols(src_attnum, tgt_attnum, ordinality)
  ON TRUE
JOIN pg_attribute AS src_att
  ON src_att.attrelid = src_tbl.oid
 AND src_att.attnum = key_cols.src_attnum
JOIN pg_attribute AS tgt_att
  ON tgt_att.attrelid = tgt_tbl.oid
 AND tgt_att.attnum = key_cols.tgt_attnum
WHERE con.contype = 'f'
  AND src_ns.nspname = ANY($1::text[])
  AND tgt_ns.nspname = ANY($1::text[])
GROUP BY
  con.conname,
  src_ns.nspname,
  src_tbl.relname,
  tgt_ns.nspname,
  tgt_tbl.relname
ORDER BY src_ns.nspname, src_tbl.relname, con.conname
"""

_PG_STATS_QUERY = """
SELECT
  schemaname AS schema_name,
  tablename AS table_name,
  attname AS column_name,
  n_distinct
FROM pg_stats
WHERE schemaname = ANY($1::text[])
ORDER BY schemaname, tablename, attname
"""


@dataclass(slots=True)
class PostgresSchemaIntrospector:
    """Discover PostgreSQL schema metadata for task generation."""

    database: DatabaseConfig
    default_visibility: Visibility
    visibility_overrides: dict[str, Visibility]

    async def introspect(self) -> SchemaGraph:
        conn = await asyncpg.connect(dsn=self.database.dsn)
        try:
            settings = control_session_settings(self.database)
            for statement in settings.timeout_sql:
                await conn.execute(statement)

            table_rows, column_rows, unique_rows, fk_rows, stats_rows = await self._fetch_metadata(conn)
            return self._build_graph(
                table_rows=table_rows,
                column_rows=column_rows,
                unique_rows=unique_rows,
                fk_rows=fk_rows,
                stats_rows=stats_rows,
            )
        finally:
            await conn.close()

    async def _fetch_metadata(self, conn: asyncpg.Connection) -> tuple[list[asyncpg.Record], ...]:
        schemas = self.database.schema_allowlist
        table_rows = await conn.fetch(_TABLE_QUERY, schemas)
        column_rows = await conn.fetch(_COLUMN_QUERY, schemas)
        unique_rows = await conn.fetch(_UNIQUE_INDEX_QUERY, schemas)
        fk_rows = await conn.fetch(_FOREIGN_KEY_QUERY, schemas)
        stats_rows = await conn.fetch(_PG_STATS_QUERY, schemas)
        return table_rows, column_rows, unique_rows, fk_rows, stats_rows

    def _build_graph(
        self,
        *,
        table_rows: list[asyncpg.Record],
        column_rows: list[asyncpg.Record],
        unique_rows: list[asyncpg.Record],
        fk_rows: list[asyncpg.Record],
        stats_rows: list[asyncpg.Record],
    ) -> SchemaGraph:
        table_profiles: dict[tuple[str, str], TableProfile] = {
            (row["schema_name"], row["table_name"]): TableProfile(
                schema_name=row["schema_name"],
                table_name=row["table_name"],
                row_estimate=int(row["row_estimate"]) if row["row_estimate"] is not None else None,
            )
            for row in table_rows
        }

        unique_constraints: dict[tuple[str, str], list[tuple[str, ...]]] = {}
        primary_keys: dict[tuple[str, str], tuple[str, ...]] = {}
        for row in unique_rows:
            table_key = (row["schema_name"], row["table_name"])
            columns = tuple(row["column_names"])
            unique_constraints.setdefault(table_key, []).append(columns)
            if row["is_primary"]:
                primary_keys[table_key] = columns

        fk_members: set[tuple[str, str, str]] = set()
        sensitivity_by_column = {
            (
                sensitivity.schema_name,
                sensitivity.table_name,
                sensitivity.column_name,
            ): sensitivity.visibility
            for sensitivity in classify_columns(
                [
                    ColumnRef(
                        schema_name=row["schema_name"],
                        table_name=row["table_name"],
                        column_name=row["column_name"],
                    )
                    for row in column_rows
                ],
                default_visibility=self.default_visibility,
                overrides=self.visibility_overrides,
            )
        }
        n_distinct_by_column = {
            (
                row["schema_name"],
                row["table_name"],
                row["column_name"],
            ): float(row["n_distinct"])
            for row in stats_rows
            if row["n_distinct"] is not None
        }

        edges: list[ForeignKeyEdge] = []
        for row in fk_rows:
            source_key = (row["source_schema"], row["source_table"])
            target_key = (row["target_schema"], row["target_table"])
            source_columns = tuple(row["source_columns"])
            target_columns = tuple(row["target_columns"])
            fk_members.update((source_key[0], source_key[1], column) for column in source_columns)
            source_table = table_profiles[source_key]
            target_table = table_profiles[target_key]
            fanout_estimate: float | None = None
            if source_table.row_estimate and target_table.row_estimate and target_table.row_estimate > 0:
                fanout_estimate = source_table.row_estimate / target_table.row_estimate
            edges.append(
                ForeignKeyEdge(
                    constraint_name=row["constraint_name"],
                    source_schema=row["source_schema"],
                    source_table=row["source_table"],
                    source_columns=source_columns,
                    target_schema=row["target_schema"],
                    target_table=row["target_table"],
                    target_columns=target_columns,
                    source_is_unique=source_columns in unique_constraints.get(source_key, []),
                    fanout_estimate=fanout_estimate,
                )
            )

        for row in column_rows:
            table_key = (row["schema_name"], row["table_name"])
            table = table_profiles[table_key]
            column_name = row["column_name"]
            table.columns.append(
                ColumnProfile(
                    schema_name=row["schema_name"],
                    table_name=row["table_name"],
                    column_name=column_name,
                    data_type=row["data_type"],
                    ordinal_position=int(row["ordinal_position"]),
                    is_nullable=bool(row["is_nullable"]),
                    visibility=sensitivity_by_column[(row["schema_name"], row["table_name"], column_name)],
                    is_primary_key=column_name in primary_keys.get(table_key, ()),
                    is_foreign_key=(row["schema_name"], row["table_name"], column_name) in fk_members,
                    is_unique=(column_name,) in unique_constraints.get(table_key, []),
                    n_distinct=n_distinct_by_column.get(
                        (row["schema_name"], row["table_name"], column_name)
                    ),
                )
            )

        for table_key, table in table_profiles.items():
            table.primary_key = primary_keys.get(table_key, ())
            table.unique_constraints = unique_constraints.get(table_key, [])
            table.columns.sort(key=lambda column: column.ordinal_position)

        return SchemaGraph(
            tables=sorted(table_profiles.values(), key=lambda table: (table.schema_name, table.table_name)),
            edges=edges,
        )
