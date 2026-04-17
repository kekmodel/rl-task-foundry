"""query primitive — JSON-spec analytic DSL over the schema.

One call expresses anything the solver could assemble as an atomic
calculus chain. The composer uses it to author canonical answers
without scripting a multi-step tool plan.

Spec shape (all fields but `from` are optional):

    {
      "from": "rental",
      "filter": [{"column": "rental_date", "op": "gt", "value": "..."}, ...],
      "join":   [{"via_edge": "rental.inventory_id->inventory"}, ...],
      "select": ["rental_id", "rental_date"],
      "sort":   [{"column": "rental_date", "direction": "desc"}],
      "limit":  10,
      "group_by":  ["customer_id"],
      "aggregate": [
          {"fn": "count", "alias": "rentals"},
          {"fn": "max", "column": "rental_date", "alias": "last"}
      ]
    }

Semantics:

- `filter` clauses apply at the `from` table (before joins).
- `join` projects through FK edges, each leaving the current target
  table at the destination of its edge.
- `select` / `sort` / `group_by` reference the current target table
  (the last join's destination, or `from` if no joins).
- `select` and `aggregate` are mutually exclusive. `group_by` is only
  valid when `aggregate` is present. `sort` may reference either
  target-table columns (no-aggregate mode) or aggregate aliases /
  group-by columns (aggregate mode).

Composer never hides behind raw SQL — the DSL is the escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_task_foundry.tooling.common.edges import (
    EdgeDirection,
    TypedEdge,
    resolve_edge,
)
from rl_task_foundry.tooling.common.schema import SchemaSnapshot, TableSpec
from rl_task_foundry.tooling.common.sql import (
    quote_ident,
    quote_table,
    readonly_select,
)
from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer._sql import (
    FilterClause,
    compile_filter_clauses,
    parse_filter_clauses,
)


_AGGREGATE_FNS: frozenset[str] = frozenset(
    ("count", "sum", "avg", "min", "max")
)
_SORT_DIRECTIONS: frozenset[str] = frozenset(("asc", "desc"))


@dataclass(frozen=True, slots=True)
class _JoinStep:
    edge: TypedEdge
    source_alias: str
    destination_alias: str


@dataclass(frozen=True, slots=True)
class _SortClause:
    reference: str
    direction: Literal["asc", "desc"]


@dataclass(frozen=True, slots=True)
class _AggregateClause:
    fn: str
    alias: str
    column: str | None


@dataclass(frozen=True, slots=True)
class _ParsedSpec:
    from_table: str
    filter_clauses: list[FilterClause]
    join_edges: list[str]
    select: list[str] | None
    sort: list[_SortClause]
    limit: int | None
    group_by: list[str]
    aggregates: list[_AggregateClause]


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{field!r} must be a string; got {type(value).__name__}"
        )
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field!r} must be an integer; got {type(value).__name__}"
        )
    return value


def _require_str_list(raw: object, field: str) -> list[str]:
    if not isinstance(raw, list):
        raise TypeError(
            f"{field!r} must be a list of strings; got {type(raw).__name__}"
        )
    out: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(
                f"{field}[{index}] must be a string; got "
                f"{type(item).__name__}"
            )
        out.append(item)
    return out


def _parse_join(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("'join' must be a list of {via_edge} entries")
    labels: list[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise TypeError(
                f"join[{index}] must be a mapping; got "
                f"{type(entry).__name__}"
            )
        label = entry.get("via_edge")
        if not isinstance(label, str):
            raise TypeError(
                f"join[{index}].via_edge must be a string"
            )
        labels.append(label)
    return labels


def _parse_sort(raw: object) -> list[_SortClause]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("'sort' must be a list of {column, direction}")
    clauses: list[_SortClause] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise TypeError(
                f"sort[{index}] must be a mapping; got "
                f"{type(entry).__name__}"
            )
        reference = entry.get("column")
        direction = entry.get("direction", "asc")
        if not isinstance(reference, str):
            raise TypeError(
                f"sort[{index}].column must be a string"
            )
        if not isinstance(direction, str) or direction not in _SORT_DIRECTIONS:
            raise ValueError(
                f"sort[{index}].direction must be 'asc' or 'desc'"
            )
        clauses.append(
            _SortClause(
                reference=reference,
                direction="asc" if direction == "asc" else "desc",
            )
        )
    return clauses


def _parse_aggregate(raw: object) -> list[_AggregateClause]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError(
            "'aggregate' must be a list of {fn, column?, alias}"
        )
    clauses: list[_AggregateClause] = []
    seen_aliases: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise TypeError(
                f"aggregate[{index}] must be a mapping; got "
                f"{type(entry).__name__}"
            )
        fn = entry.get("fn")
        alias = entry.get("alias")
        column = entry.get("column")
        if not isinstance(fn, str) or fn not in _AGGREGATE_FNS:
            raise ValueError(
                f"aggregate[{index}].fn must be one of "
                f"{sorted(_AGGREGATE_FNS)}; got {fn!r}"
            )
        if not isinstance(alias, str):
            raise TypeError(
                f"aggregate[{index}].alias must be a string"
            )
        if alias in seen_aliases:
            raise ValueError(
                f"aggregate aliases must be unique; duplicate {alias!r}"
            )
        seen_aliases.add(alias)
        if fn == "count":
            if column is not None and not isinstance(column, str):
                raise TypeError(
                    f"aggregate[{index}].column must be null or a string"
                )
            clauses.append(
                _AggregateClause(
                    fn="count",
                    alias=alias,
                    column=column if isinstance(column, str) else None,
                )
            )
            continue
        if not isinstance(column, str):
            raise TypeError(
                f"aggregate[{index}].column is required for fn={fn!r}"
            )
        clauses.append(
            _AggregateClause(fn=fn, alias=alias, column=column)
        )
    return clauses


def _parse_spec(spec: object) -> _ParsedSpec:
    if not isinstance(spec, dict):
        raise TypeError("query spec must be a mapping")
    from_table = spec.get("from")
    if not isinstance(from_table, str):
        raise TypeError("query spec must include 'from' as a string")
    raw_filter = spec.get("filter")
    if raw_filter is None:
        filter_clauses = parse_filter_clauses(None)
    elif isinstance(raw_filter, list):
        filter_clauses = parse_filter_clauses(raw_filter)
    else:
        raise TypeError("'filter' must be a list of {column, op, value}")
    join_edges = _parse_join(spec.get("join"))
    raw_select = spec.get("select")
    select = _require_str_list(raw_select, "select") if raw_select is not None else None
    sort = _parse_sort(spec.get("sort"))
    raw_limit = spec.get("limit")
    limit = _require_int(raw_limit, "limit") if raw_limit is not None else None
    if limit is not None and limit <= 0:
        raise ValueError("'limit' must be a positive integer")
    raw_group_by = spec.get("group_by")
    group_by = (
        _require_str_list(raw_group_by, "group_by")
        if raw_group_by is not None
        else []
    )
    aggregates = _parse_aggregate(spec.get("aggregate"))
    if aggregates and select is not None:
        raise ValueError(
            "'select' and 'aggregate' are mutually exclusive"
        )
    if group_by and not aggregates:
        raise ValueError("'group_by' requires at least one aggregate")
    return _ParsedSpec(
        from_table=from_table,
        filter_clauses=filter_clauses,
        join_edges=join_edges,
        select=select,
        sort=sort,
        limit=limit,
        group_by=group_by,
        aggregates=aggregates,
    )


def _resolve_join_chain(
    snapshot: SchemaSnapshot,
    from_table: str,
    join_labels: list[str],
) -> list[_JoinStep]:
    steps: list[_JoinStep] = []
    current_table = from_table
    current_alias_index = 0
    for index, label in enumerate(join_labels):
        edge = resolve_edge(snapshot, current_table, label)
        steps.append(
            _JoinStep(
                edge=edge,
                source_alias=f"t{current_alias_index}",
                destination_alias=f"t{current_alias_index + 1}",
            )
        )
        current_table = edge.destination_table
        current_alias_index += 1
    return steps


def _compile_join_chain(
    snapshot: SchemaSnapshot,
    steps: list[_JoinStep],
) -> str:
    fragments: list[str] = []
    for step in steps:
        edge = step.edge
        if edge.direction is EdgeDirection.FORWARD:
            source_column = edge.spec.source_column
            destination_column = edge.spec.target_column
        else:
            source_column = edge.spec.target_column
            destination_column = edge.spec.source_column
        destination_spec = snapshot.table(edge.destination_table)
        fragments.append(
            f"JOIN "
            f"{quote_table(destination_spec.schema, destination_spec.name)} "
            f"AS {step.destination_alias} "
            f"ON {step.destination_alias}.{quote_ident(destination_column)} "
            f"= {step.source_alias}.{quote_ident(source_column)}"
        )
    return " ".join(fragments)


def _compile_aggregate_expr(
    aggregate: _AggregateClause, current_alias: str, target: TableSpec
) -> str:
    if aggregate.fn == "count":
        if aggregate.column is None:
            return f"COUNT(*) AS {quote_ident(aggregate.alias)}"
        target.column(aggregate.column)
        return (
            f"COUNT({current_alias}.{quote_ident(aggregate.column)}) "
            f"AS {quote_ident(aggregate.alias)}"
        )
    if aggregate.column is None:
        raise ValueError(
            f"aggregate fn={aggregate.fn!r} requires a column"
        )
    target.column(aggregate.column)
    return (
        f"{aggregate.fn.upper()}({current_alias}.{quote_ident(aggregate.column)}) "
        f"AS {quote_ident(aggregate.alias)}"
    )


def _compile_sort(
    clause: _SortClause,
    *,
    current_alias: str,
    target: TableSpec,
    allowed_aliases: frozenset[str],
) -> str:
    direction = clause.direction.upper()
    if clause.reference in allowed_aliases:
        return f"{quote_ident(clause.reference)} {direction}"
    target.column(clause.reference)
    return f"{current_alias}.{quote_ident(clause.reference)} {direction}"


async def query(
    session: ComposerSession,
    *,
    spec: dict[str, object],
) -> dict[str, object]:
    """Execute a query-DSL spec and return rows + column list."""
    parsed = _parse_spec(spec)
    snapshot = session.snapshot
    from_spec = snapshot.table(parsed.from_table)
    steps = _resolve_join_chain(snapshot, parsed.from_table, parsed.join_edges)
    final_alias = steps[-1].destination_alias if steps else "t0"
    target_spec = (
        snapshot.table(steps[-1].edge.destination_table)
        if steps
        else from_spec
    )
    # Validate group_by columns on the target
    for column_name in parsed.group_by:
        target_spec.column(column_name)
    clause_sql, params, next_index = compile_filter_clauses(
        table_spec=from_spec,
        alias="t0",
        clauses=parsed.filter_clauses,
        start_param=1,
    )
    where_sql = f"WHERE {clause_sql}" if clause_sql else ""

    # Figure out SELECT columns/expressions and which columns the caller
    # sees in the result dicts.
    select_fragments: list[str] = []
    output_columns: list[str] = []
    alias_set: set[str] = set()
    if parsed.aggregates:
        for group_column in parsed.group_by:
            select_fragments.append(
                f"{final_alias}.{quote_ident(group_column)} AS "
                f"{quote_ident(group_column)}"
            )
            output_columns.append(group_column)
        for aggregate in parsed.aggregates:
            select_fragments.append(
                _compile_aggregate_expr(aggregate, final_alias, target_spec)
            )
            output_columns.append(aggregate.alias)
            alias_set.add(aggregate.alias)
    elif parsed.select is not None:
        for column_name in parsed.select:
            target_spec.column(column_name)
            select_fragments.append(
                f"{final_alias}.{quote_ident(column_name)} AS "
                f"{quote_ident(column_name)}"
            )
            output_columns.append(column_name)
    else:
        for column in target_spec.columns:
            select_fragments.append(
                f"{final_alias}.{quote_ident(column.name)} AS "
                f"{quote_ident(column.name)}"
            )
            output_columns.append(column.name)

    group_by_sql = ""
    if parsed.group_by:
        group_by_sql = "GROUP BY " + ", ".join(
            f"{final_alias}.{quote_ident(column)}"
            for column in parsed.group_by
        )

    allowed_refs = frozenset(alias_set | set(parsed.group_by))
    sort_fragments: list[str] = []
    for sort_clause in parsed.sort:
        sort_fragments.append(
            _compile_sort(
                sort_clause,
                current_alias=final_alias,
                target=target_spec,
                allowed_aliases=allowed_refs,
            )
        )
    order_sql = (
        "ORDER BY " + ", ".join(sort_fragments) if sort_fragments else ""
    )
    limit_sql = f"LIMIT {int(parsed.limit)}" if parsed.limit is not None else ""

    join_sql = _compile_join_chain(snapshot, steps)
    sql = readonly_select(
        f"SELECT {', '.join(select_fragments)} "
        f"FROM {quote_table(from_spec.schema, from_spec.name)} AS t0 "
        f"{join_sql} "
        f"{where_sql} "
        f"{group_by_sql} "
        f"{order_sql} "
        f"{limit_sql}"
    )

    rows = await session.connection.fetch(sql, *params)
    materialized = [
        {column: row[column] for column in output_columns} for row in rows
    ]
    return {
        "columns": output_columns,
        "rows": materialized,
        "row_count": len(materialized),
    }


__all__ = ["query"]
