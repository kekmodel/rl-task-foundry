"""neighborhood primitive — anchor-rooted entity graph.

Given a row identified by (`table`, `row_id`) the neighborhood call
returns the anchor attributes plus, for every relationship edge originating at
that table, a bounded sample of connected row IDs and the total count
on that edge.

Forward edges (many-to-one) resolve through the anchor's FK column to a
single destination row (or zero if the FK is NULL). Reverse edges
(one-to-many) run a sample + count against the source table keyed on
the anchor's target-column value.

Only depth=1 is supported in this round; deeper traversal lands with
the composer tool_factory integration.
"""

from __future__ import annotations

from rl_task_foundry.tooling.common.edges import (
    EdgeDirection,
    TypedEdge,
    available_edges,
)
from rl_task_foundry.tooling.common.schema import SchemaSnapshot, TableSpec
from rl_task_foundry.tooling.common.sql import (
    quote_ident,
    quote_table,
    readonly_select,
)
from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer._sql import (
    coerce_asyncpg_int,
)


def _pk_where_clause(
    *,
    alias: str,
    pk_columns: tuple[str, ...],
    row_id: object,
) -> tuple[str, tuple[object, ...]]:
    if not pk_columns:
        raise ValueError("neighborhood requires the anchor table to have a primary key")
    if len(pk_columns) == 1:
        return f"{alias}.{quote_ident(pk_columns[0])} = $1", (row_id,)
    if not isinstance(row_id, list | tuple):
        raise TypeError(
            "row_id must be an array when the anchor table has a composite "
            "primary key"
        )
    if len(row_id) != len(pk_columns):
        raise ValueError(
            f"row_id must have {len(pk_columns)} values for this composite "
            "primary key"
        )
    parts = [
        f"{alias}.{quote_ident(column)} = ${index}"
        for index, column in enumerate(pk_columns, start=1)
    ]
    return " AND ".join(parts), tuple(row_id)


def _pk_select_parts(table_spec: TableSpec, alias: str) -> list[str]:
    return [
        f"{alias}.{quote_ident(column)} AS __pk_{index}"
        for index, column in enumerate(table_spec.primary_key)
    ]


def _pk_order_by(table_spec: TableSpec, alias: str) -> str:
    return ", ".join(
        f"{alias}.{quote_ident(column)} ASC"
        for column in table_spec.primary_key
    )


def _record_id_from_row(row: object, pk_columns: tuple[str, ...]) -> object:
    if len(pk_columns) == 1:
        try:
            return row["__pk_0"]  # type: ignore[index]
        except KeyError:
            return row["id"]  # type: ignore[index]
    values = [
        row[f"__pk_{index}"]  # type: ignore[index]
        for index, _column in enumerate(pk_columns)
    ]
    if len(values) == 1:
        return values[0]
    return values


def _edge_values(
    anchor_attrs: dict[str, object],
    columns: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(anchor_attrs.get(column) for column in columns)


def _edge_where_clause(
    *,
    alias: str,
    columns: tuple[str, ...],
) -> str:
    return " AND ".join(
        f"{alias}.{quote_ident(column)} = ${index}"
        for index, column in enumerate(columns, start=1)
    )


def _normalise_edge_id(values: tuple[object, ...]) -> object:
    if len(values) == 1:
        return values[0]
    return list(values)


async def _fetch_anchor_attributes(
    session: ComposerSession,
    table_spec: TableSpec,
    pk_columns: tuple[str, ...],
    row_id: object,
) -> dict[str, object]:
    column_names = [column.name for column in table_spec.exposed_columns]
    select_parts = ", ".join(f"t.{quote_ident(name)}" for name in column_names)
    where_clause, params = _pk_where_clause(
        alias="t",
        pk_columns=pk_columns,
        row_id=row_id,
    )
    sql = readonly_select(
        f"SELECT {select_parts} "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS t "
        f"WHERE {where_clause} "
        f"LIMIT 1"
    )
    row = await session.connection.fetchrow(sql, *params)
    if row is None:
        raise LookupError(
            f"no row with primary key {row_id!r} in table "
            f"{table_spec.handle!r}"
        )
    return {name: row[name] for name in column_names}


async def _forward_edge_payload(
    session: ComposerSession,
    snapshot: SchemaSnapshot,
    edge: TypedEdge,
    anchor_attrs: dict[str, object],
    max_per_edge: int,
) -> dict[str, object]:
    source_values = _edge_values(anchor_attrs, edge.spec.source_columns)
    dest_spec = snapshot.table(edge.destination_table)
    dest_columns = edge.spec.target_columns
    base_payload: dict[str, object] = {
        "label": edge.label,
        "direction": edge.direction.value,
        "destination_table": edge.destination_table,
    }
    if not dest_spec.primary_key:
        return {
            **base_payload,
            "unsupported": True,
            "reason": "destination has no primary key",
        }
    if any(value is None for value in source_values):
        return {**base_payload, "total_count": 0, "sample_ids": []}
    if dest_columns == dest_spec.primary_key:
        return {
            **base_payload,
            "total_count": 1,
            "sample_ids": [_normalise_edge_id(source_values)],
        }
    select_parts = ", ".join(_pk_select_parts(dest_spec, "dst"))
    where_clause = _edge_where_clause(alias="dst", columns=dest_columns)
    sample_sql = readonly_select(
        f"SELECT {select_parts} "
        f"FROM {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"WHERE {where_clause} "
        f"ORDER BY {_pk_order_by(dest_spec, 'dst')} "
        f"LIMIT {int(max_per_edge)}"
    )
    sample_rows = await session.connection.fetch(sample_sql, *source_values)
    sample_ids = [
        _record_id_from_row(row, dest_spec.primary_key) for row in sample_rows
    ]
    count_sql = readonly_select(
        f"SELECT COUNT(*) AS cnt "
        f"FROM {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"WHERE {where_clause}"
    )
    total_value = await session.connection.fetchval(count_sql, *source_values)
    return {
        **base_payload,
        "total_count": coerce_asyncpg_int(total_value),
        "sample_ids": sample_ids,
    }


async def _reverse_edge_payload(
    session: ComposerSession,
    snapshot: SchemaSnapshot,
    edge: TypedEdge,
    anchor_attrs: dict[str, object],
    max_per_edge: int,
) -> dict[str, object]:
    anchor_values = _edge_values(anchor_attrs, edge.spec.target_columns)
    dest_spec = snapshot.table(edge.destination_table)
    dest_match_columns = edge.spec.source_columns
    base_payload: dict[str, object] = {
        "label": edge.label,
        "direction": edge.direction.value,
        "destination_table": edge.destination_table,
    }
    if not dest_spec.primary_key:
        return {
            **base_payload,
            "unsupported": True,
            "reason": "destination has no primary key",
        }
    if any(value is None for value in anchor_values):
        return {**base_payload, "total_count": 0, "sample_ids": []}
    select_parts = ", ".join(_pk_select_parts(dest_spec, "dst"))
    where_clause = _edge_where_clause(alias="dst", columns=dest_match_columns)
    sample_sql = readonly_select(
        f"SELECT {select_parts} "
        f"FROM {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"WHERE {where_clause} "
        f"ORDER BY {_pk_order_by(dest_spec, 'dst')} "
        f"LIMIT {int(max_per_edge)}"
    )
    sample_rows = await session.connection.fetch(sample_sql, *anchor_values)
    sample_ids = [
        _record_id_from_row(row, dest_spec.primary_key) for row in sample_rows
    ]
    count_sql = readonly_select(
        f"SELECT COUNT(*) AS cnt "
        f"FROM {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"WHERE {where_clause}"
    )
    total_value = await session.connection.fetchval(count_sql, *anchor_values)
    return {
        **base_payload,
        "total_count": coerce_asyncpg_int(total_value),
        "sample_ids": sample_ids,
    }


async def neighborhood(
    session: ComposerSession,
    *,
    table: str,
    row_id: object,
    depth: int = 1,
    max_per_edge: int = 5,
) -> dict[str, object]:
    """Anchor-rooted view: attributes + per-edge sample IDs + counts."""
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    if depth != 1:
        raise NotImplementedError(
            "neighborhood currently supports depth=1 only"
        )
    if isinstance(max_per_edge, bool) or not isinstance(max_per_edge, int):
        raise TypeError("max_per_edge must be an integer")
    if max_per_edge < 1:
        raise ValueError("max_per_edge must be a positive integer")
    snapshot = session.snapshot
    table_spec = snapshot.table(table)
    anchor_attrs = await _fetch_anchor_attributes(
        session, table_spec, table_spec.primary_key, row_id
    )
    edges_payload: list[dict[str, object]] = []
    for edge in available_edges(snapshot, table_spec.handle):
        if edge.direction is EdgeDirection.FORWARD:
            edges_payload.append(
                await _forward_edge_payload(
                    session, snapshot, edge, anchor_attrs, max_per_edge
                )
            )
        else:
            edges_payload.append(
                await _reverse_edge_payload(
                    session, snapshot, edge, anchor_attrs, max_per_edge
                )
            )
    return {
        "anchor": {
            "table": table_spec.handle,
            "row_id": row_id,
            "attributes": anchor_attrs,
        },
        "edges": edges_payload,
    }


__all__ = ["neighborhood"]
