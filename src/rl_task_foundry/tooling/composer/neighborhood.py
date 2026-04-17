"""neighborhood primitive — anchor-rooted entity graph.

Given a row identified by (`table`, `row_id`) the neighborhood call
returns the anchor attributes plus, for every FK edge originating at
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


def _single_column_pk(table_spec: TableSpec) -> str:
    if len(table_spec.primary_key) != 1:
        raise NotImplementedError(
            f"neighborhood supports single-column primary keys only; "
            f"{table_spec.qualified_name!r} has a composite PK"
        )
    return table_spec.primary_key[0]


async def _fetch_anchor_attributes(
    session: ComposerSession,
    table_spec: TableSpec,
    pk: str,
    row_id: object,
) -> dict[str, object]:
    column_names = [column.name for column in table_spec.columns]
    select_parts = ", ".join(f"t.{quote_ident(name)}" for name in column_names)
    sql = readonly_select(
        f"SELECT {select_parts} "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS t "
        f"WHERE t.{quote_ident(pk)} = $1 "
        f"LIMIT 1"
    )
    row = await session.connection.fetchrow(sql, row_id)
    if row is None:
        raise LookupError(
            f"no row with primary key {row_id!r} in table "
            f"{table_spec.name!r}"
        )
    return {name: row[name] for name in column_names}


async def _forward_edge_payload(
    session: ComposerSession,
    snapshot: SchemaSnapshot,
    edge: TypedEdge,
    anchor_attrs: dict[str, object],
    max_per_edge: int,
) -> dict[str, object]:
    source_column = edge.spec.source_column
    source_value = anchor_attrs.get(source_column)
    dest_spec = snapshot.table(edge.destination_table)
    dest_column = edge.spec.target_column
    base_payload: dict[str, object] = {
        "label": edge.label,
        "direction": edge.direction.value,
        "destination_table": edge.destination_table,
    }
    if len(dest_spec.primary_key) != 1:
        return {
            **base_payload,
            "unsupported": True,
            "reason": "destination has no single-column primary key",
        }
    dest_pk = dest_spec.primary_key[0]
    if source_value is None:
        return {**base_payload, "total_count": 0, "sample_ids": []}
    if dest_column == dest_pk:
        return {
            **base_payload,
            "total_count": 1,
            "sample_ids": [source_value],
        }
    sample_sql = readonly_select(
        f"SELECT dst.{quote_ident(dest_pk)} AS id "
        f"FROM {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"WHERE dst.{quote_ident(dest_column)} = $1 "
        f"ORDER BY dst.{quote_ident(dest_pk)} ASC "
        f"LIMIT {int(max_per_edge)}"
    )
    sample_rows = await session.connection.fetch(sample_sql, source_value)
    sample_ids = [row["id"] for row in sample_rows]
    count_sql = readonly_select(
        f"SELECT COUNT(*) AS cnt "
        f"FROM {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"WHERE dst.{quote_ident(dest_column)} = $1"
    )
    total_value = await session.connection.fetchval(count_sql, source_value)
    return {
        **base_payload,
        "total_count": _as_int(total_value),
        "sample_ids": sample_ids,
    }


async def _reverse_edge_payload(
    session: ComposerSession,
    snapshot: SchemaSnapshot,
    edge: TypedEdge,
    anchor_attrs: dict[str, object],
    max_per_edge: int,
) -> dict[str, object]:
    anchor_match_column = edge.spec.target_column  # on origin (= anchor's table)
    anchor_value = anchor_attrs.get(anchor_match_column)
    dest_spec = snapshot.table(edge.destination_table)
    dest_match_column = edge.spec.source_column  # on destination (= source table)
    base_payload: dict[str, object] = {
        "label": edge.label,
        "direction": edge.direction.value,
        "destination_table": edge.destination_table,
    }
    if len(dest_spec.primary_key) != 1:
        return {
            **base_payload,
            "unsupported": True,
            "reason": "destination has no single-column primary key",
        }
    dest_pk = dest_spec.primary_key[0]
    if anchor_value is None:
        return {**base_payload, "total_count": 0, "sample_ids": []}
    sample_sql = readonly_select(
        f"SELECT dst.{quote_ident(dest_pk)} AS id "
        f"FROM {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"WHERE dst.{quote_ident(dest_match_column)} = $1 "
        f"ORDER BY dst.{quote_ident(dest_pk)} ASC "
        f"LIMIT {int(max_per_edge)}"
    )
    sample_rows = await session.connection.fetch(sample_sql, anchor_value)
    sample_ids = [row["id"] for row in sample_rows]
    count_sql = readonly_select(
        f"SELECT COUNT(*) AS cnt "
        f"FROM {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"WHERE dst.{quote_ident(dest_match_column)} = $1"
    )
    total_value = await session.connection.fetchval(count_sql, anchor_value)
    return {
        **base_payload,
        "total_count": _as_int(total_value),
        "sample_ids": sample_ids,
    }


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return int(str(value))
    return value


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
    pk = _single_column_pk(table_spec)
    anchor_attrs = await _fetch_anchor_attributes(
        session, table_spec, pk, row_id
    )
    edges_payload: list[dict[str, object]] = []
    for edge in available_edges(snapshot, table):
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
            "table": table_spec.name,
            "row_id": row_id,
            "attributes": anchor_attrs,
        },
        "edges": edges_payload,
    }


__all__ = ["neighborhood"]
