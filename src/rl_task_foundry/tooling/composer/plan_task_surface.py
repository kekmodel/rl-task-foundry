"""plan_task_surface primitive -- structural candidate surfaces.

This composer-only primitive starts from an observed anchor row and returns
direct or parent-to-sibling, primary-key-backed record surfaces that look
structurally suitable for authoring. It does not write the final task, request,
topic, answer contract, or label. Final label evidence still comes from
``query``.
"""

from __future__ import annotations

from rl_task_foundry.infra.visibility import VISIBILITY_USER_VISIBLE
from rl_task_foundry.tooling.common.edges import (
    EdgeDirection,
    TypedEdge,
    available_edges,
)
from rl_task_foundry.tooling.common.schema import ColumnSpec, TableSpec
from rl_task_foundry.tooling.common.sql import (
    quote_ident,
    quote_table,
    readonly_select,
)
from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer._sql import coerce_asyncpg_int
from rl_task_foundry.tooling.composer.neighborhood import (
    _edge_values,
    _edge_where_clause,
    _fetch_anchor_attributes,
    _pk_order_by,
    _pk_select_parts,
)

_ORDERABLE_TYPES = frozenset(
    {
        "bigint",
        "bigserial",
        "date",
        "decimal",
        "double precision",
        "float4",
        "float8",
        "int2",
        "int4",
        "int8",
        "integer",
        "money",
        "numeric",
        "real",
        "serial",
        "smallint",
        "smallserial",
        "time",
        "time with time zone",
        "time without time zone",
        "timestamp",
        "timestamp with time zone",
        "timestamp without time zone",
        "timestamptz",
    }
)


def _normalise_data_type(data_type: str) -> str:
    return data_type.strip().lower()


def _candidate_output_columns(table_spec: TableSpec) -> list[ColumnSpec]:
    return [
        column
        for column in table_spec.exposed_columns
        if column.visibility == VISIBILITY_USER_VISIBLE and not column.is_handle_column
    ]


def _candidate_order_columns(table_spec: TableSpec) -> list[ColumnSpec]:
    return [
        column
        for column in _candidate_output_columns(table_spec)
        if _normalise_data_type(column.data_type) in _ORDERABLE_TYPES
    ]


def _edge_match(
    edge: TypedEdge,
    anchor_attrs: dict[str, object],
) -> tuple[tuple[str, ...], tuple[object, ...]]:
    if edge.direction is EdgeDirection.FORWARD:
        return edge.spec.target_columns, _edge_values(
            anchor_attrs, edge.spec.source_columns
        )
    return edge.spec.source_columns, _edge_values(
        anchor_attrs, edge.spec.target_columns
    )


async def _count_matching_rows(
    session: ComposerSession,
    table_spec: TableSpec,
    match_columns: tuple[str, ...],
    match_values: tuple[object, ...],
) -> int:
    where_clause = _edge_where_clause(alias="dst", columns=match_columns)
    count_sql = readonly_select(
        f"SELECT COUNT(*) AS cnt "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS dst "
        f"WHERE {where_clause}"
    )
    total_value = await session.connection.fetchval(count_sql, *match_values)
    return coerce_asyncpg_int(total_value)


async def _sample_matching_rows(
    session: ComposerSession,
    table_spec: TableSpec,
    match_columns: tuple[str, ...],
    match_values: tuple[object, ...],
    output_columns: list[ColumnSpec],
    max_sample_rows: int,
) -> list[dict[str, object]]:
    select_parts = _pk_select_parts(table_spec, "dst")
    select_parts.extend(
        f"dst.{quote_ident(column.name)} AS {quote_ident(column.name)}"
        for column in output_columns
    )
    where_clause = _edge_where_clause(alias="dst", columns=match_columns)
    sample_sql = readonly_select(
        f"SELECT {', '.join(select_parts)} "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS dst "
        f"WHERE {where_clause} "
        f"ORDER BY {_pk_order_by(table_spec, 'dst')} "
        f"LIMIT {int(max_sample_rows)}"
    )
    rows = await session.connection.fetch(sample_sql, *match_values)
    output_names = [column.name for column in output_columns]
    return [
        {name: row[name] for name in output_names}
        for row in rows
    ]


async def _fetch_matching_attributes(
    session: ComposerSession,
    table_spec: TableSpec,
    match_columns: tuple[str, ...],
    match_values: tuple[object, ...],
) -> dict[str, object] | None:
    column_names = [column.name for column in table_spec.exposed_columns]
    select_parts = ", ".join(f"t.{quote_ident(name)}" for name in column_names)
    where_clause = _edge_where_clause(alias="t", columns=match_columns)
    sql = readonly_select(
        f"SELECT {select_parts} "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS t "
        f"WHERE {where_clause} "
        f"LIMIT 1"
    )
    row = await session.connection.fetchrow(sql, *match_values)
    if row is None:
        return None
    return {name: row[name] for name in column_names}


def _output_payloads(
    output_columns: list[ColumnSpec],
    sample_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    sample_size = len(sample_rows)
    payloads: list[dict[str, object]] = []
    for column in output_columns:
        non_null = sum(
            1 for row in sample_rows if row.get(column.name) is not None
        )
        payloads.append(
            {
                "field": column.name,
                "data_type": column.data_type,
                "nullable": column.is_nullable,
                "visibility": column.visibility,
                "sample_size": sample_size,
                "sample_non_null": non_null,
                "selected_by": "user_visible_non_handle",
            }
        )
    return payloads


def _order_payloads(order_columns: list[ColumnSpec], total_count: int) -> list[dict[str, object]]:
    return [
        {
            "field": column.name,
            "data_type": column.data_type,
            "nullable": column.is_nullable,
            "directions": ["asc", "desc"],
            "requires_user_phrase": True,
            "tie_check_required": total_count > 1,
        }
        for column in order_columns
    ]


def _structural_risks(
    *,
    primary_key_backed: bool,
    total_count: int,
    output_count: int,
    order_count: int,
) -> list[str]:
    risks: list[str] = []
    if not primary_key_backed:
        risks.append("not_primary_key_backed")
    if total_count == 0:
        risks.append("zero_related_rows")
    elif total_count == 1:
        risks.append("single_related_row")
    elif total_count > 5:
        risks.append("many_related_rows_need_order_limit_or_aggregate")
    if output_count == 0:
        risks.append("no_user_visible_non_handle_outputs")
    if total_count > 1 and order_count == 0:
        risks.append("no_structural_order_candidates")
    if total_count > 1 and order_count > 0:
        risks.append("order_ties_must_be_checked_by_final_query")
    return risks


def _candidate_score(
    *,
    primary_key_backed: bool,
    total_count: int,
    output_count: int,
    order_count: int,
) -> int:
    score = 0
    if primary_key_backed:
        score += 100
    if 3 <= total_count <= 5:
        score += 50
    elif total_count > 5:
        score += 30
    elif total_count == 2:
        score += 20
    elif total_count == 1:
        score += 5
    score += min(output_count, 4) * 5
    score += min(order_count, 2) * 5
    if output_count == 0:
        score -= 20
    if total_count == 0:
        score -= 30
    return score


async def _surface_candidate_from_match(
    session: ComposerSession,
    *,
    surface_path: list[dict[str, object]],
    destination_table: str,
    relationship_direction: str,
    match_columns: tuple[str, ...],
    match_values: tuple[object, ...],
    path_kind: str,
    max_sample_rows: int,
) -> dict[str, object]:
    destination = session.snapshot.table(destination_table)
    output_columns = _candidate_output_columns(destination)
    order_columns = _candidate_order_columns(destination)
    primary_key_backed = bool(destination.primary_key)

    if any(value is None for value in match_values):
        total_count = 0
        sample_rows: list[dict[str, object]] = []
    elif primary_key_backed:
        total_count = await _count_matching_rows(
            session,
            destination,
            match_columns,
            match_values,
        )
        sample_rows = await _sample_matching_rows(
            session,
            destination,
            match_columns,
            match_values,
            output_columns,
            max_sample_rows,
        )
    else:
        total_count = await _count_matching_rows(
            session,
            destination,
            match_columns,
            match_values,
        )
        sample_rows = []

    output_payloads = _output_payloads(output_columns, sample_rows)
    order_payloads = _order_payloads(order_columns, total_count)
    risks = _structural_risks(
        primary_key_backed=primary_key_backed,
        total_count=total_count,
        output_count=len(output_payloads),
        order_count=len(order_payloads),
    )
    return {
        "score": _candidate_score(
            primary_key_backed=primary_key_backed,
            total_count=total_count,
            output_count=len(output_payloads),
            order_count=len(order_payloads),
        ),
        "surface_path": surface_path,
        "record_surface": {
            "table": destination_table,
            "primary_key_backed": primary_key_backed,
            "record_revisitable": primary_key_backed,
            "relationship_direction": relationship_direction,
            "path_kind": path_kind,
            "match_columns": list(match_columns),
            "total_count": total_count,
            "sample_size": len(sample_rows),
        },
        "candidate_outputs": output_payloads,
        "candidate_orders": order_payloads,
        "structural_risks": risks,
        "next_step": (
            "Use query(spec) to prove the final rows and diagnostics; "
            "this planning candidate is not label evidence."
        ),
    }


async def _surface_candidate(
    session: ComposerSession,
    *,
    anchor_table: str,
    edge: TypedEdge,
    anchor_attrs: dict[str, object],
    max_sample_rows: int,
) -> dict[str, object]:
    match_columns, match_values = _edge_match(edge, anchor_attrs)
    return await _surface_candidate_from_match(
        session,
        surface_path=[
            {
                "from_table": anchor_table,
                "via_edge": edge.label,
                "direction": edge.direction.value,
                "to_table": edge.destination_table,
            }
        ],
        destination_table=edge.destination_table,
        relationship_direction=edge.direction.value,
        path_kind="direct",
        match_columns=match_columns,
        match_values=match_values,
        max_sample_rows=max_sample_rows,
    )


async def _parent_to_sibling_candidates(
    session: ComposerSession,
    *,
    anchor_table: str,
    edge: TypedEdge,
    anchor_attrs: dict[str, object],
    max_sample_rows: int,
) -> list[dict[str, object]]:
    if edge.direction is not EdgeDirection.FORWARD:
        return []
    parent = session.snapshot.table(edge.destination_table)
    parent_match_columns, parent_match_values = _edge_match(edge, anchor_attrs)
    if any(value is None for value in parent_match_values):
        return []
    parent_attrs = await _fetch_matching_attributes(
        session,
        parent,
        parent_match_columns,
        parent_match_values,
    )
    if parent_attrs is None:
        return []

    candidates: list[dict[str, object]] = []
    first_step = {
        "from_table": anchor_table,
        "via_edge": edge.label,
        "direction": edge.direction.value,
        "to_table": edge.destination_table,
    }
    for sibling_edge in available_edges(session.snapshot, parent.handle):
        if sibling_edge.direction is not EdgeDirection.REVERSE:
            continue
        sibling_match_columns, sibling_match_values = _edge_match(
            sibling_edge,
            parent_attrs,
        )
        candidates.append(
            await _surface_candidate_from_match(
                session,
                surface_path=[
                    first_step,
                    {
                        "from_table": parent.handle,
                        "via_edge": sibling_edge.label,
                        "direction": sibling_edge.direction.value,
                        "to_table": sibling_edge.destination_table,
                    },
                ],
                destination_table=sibling_edge.destination_table,
                relationship_direction=sibling_edge.direction.value,
                path_kind="via_parent",
                match_columns=sibling_match_columns,
                match_values=sibling_match_values,
                max_sample_rows=max_sample_rows,
            )
        )
    return candidates


async def plan_task_surface(
    session: ComposerSession,
    *,
    table: str,
    row_id: object,
    max_candidates: int = 5,
    max_sample_rows: int = 5,
) -> dict[str, object]:
    """Return structural task-surface candidates for an anchor row."""
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise TypeError("max_candidates must be an integer")
    if max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if isinstance(max_sample_rows, bool) or not isinstance(max_sample_rows, int):
        raise TypeError("max_sample_rows must be an integer")
    if max_sample_rows < 1:
        raise ValueError("max_sample_rows must be a positive integer")

    snapshot = session.snapshot
    table_spec = snapshot.table(table)
    anchor_attrs = await _fetch_anchor_attributes(
        session,
        table_spec,
        table_spec.primary_key,
        row_id,
    )
    candidates: list[dict[str, object]] = []
    for edge in available_edges(snapshot, table_spec.handle):
        candidates.append(
            await _surface_candidate(
                session,
                anchor_table=table_spec.handle,
                edge=edge,
                anchor_attrs=anchor_attrs,
                max_sample_rows=max_sample_rows,
            )
        )
        candidates.extend(
            await _parent_to_sibling_candidates(
                session,
                anchor_table=table_spec.handle,
                edge=edge,
                anchor_attrs=anchor_attrs,
                max_sample_rows=max_sample_rows,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            -int(candidate["score"]),
            str(candidate["record_surface"]["table"]),  # type: ignore[index]
            str(candidate["surface_path"][0]["via_edge"]),  # type: ignore[index]
        )
    )
    limited = candidates[:max_candidates]
    return {
        "anchor": {
            "table": table_spec.handle,
            "row_id": row_id,
        },
        "candidate_count": len(limited),
        "candidates": limited,
        "notes": [
            "planning_only",
            "not_label_evidence",
            "final_query_required",
        ],
    }


__all__ = ["plan_task_surface"]
