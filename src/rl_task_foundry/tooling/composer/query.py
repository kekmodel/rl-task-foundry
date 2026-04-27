"""query primitive: alias-qualified authoring DSL over the schema.

The synthesis authoring path uses this tool for canonical-answer accuracy:
one call can express a typed join chain, chain-aware filters, projections,
ordering, and aggregates.

Spec shape:

    {
      "from": {"table": "<anchor_table>", "as": "a"},
      "join": [
        {"from": "a", "via_edge": "<anchor_table><-<link_table>.<anchor_fk>", "as": "l"},
        {"from": "l", "via_edge": "<link_table><-<event_table>.<link_fk>", "as": "e"}
      ],
      "where": [
        {"ref": {"as": "a", "column": "<anchor_pk>"}, "op": "eq", "value": 128},
        {"ref": {"as": "e", "column": "<event_time>"}, "op": "gte", "value": "2026-01-01"}
      ],
      "select": [
        {"ref": {"as": "e", "column": "<event_time>"}, "as": "event_time"}
      ],
      "order_by": [
        {"ref": {"as": "e", "column": "<event_time>"}, "direction": "asc"}
      ],
      "limit": 5
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_task_foundry.tooling.common.edges import TypedEdge, resolve_edge
from rl_task_foundry.tooling.common.payload import ensure_int as _require_int
from rl_task_foundry.tooling.common.schema import TableSpec
from rl_task_foundry.tooling.common.sql import (
    coerce_param,
    coerce_scalar,
    quote_ident,
    quote_table,
    readonly_select,
)
from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer._sql import FILTER_OPS, array_cast_for

_AGGREGATE_FNS: frozenset[str] = frozenset(
    ("count", "sum", "avg", "min", "max")
)
_SORT_DIRECTIONS: frozenset[str] = frozenset(("asc", "desc"))


@dataclass(frozen=True, slots=True)
class _RawRef:
    alias: str
    column: str


@dataclass(frozen=True, slots=True)
class _ResolvedRef:
    column: str
    table: TableSpec
    sql_alias: str
    user_alias: str


@dataclass(frozen=True, slots=True)
class _FromSpec:
    table: str
    alias: str


@dataclass(frozen=True, slots=True)
class _JoinSpec:
    source_alias: str | None
    edge_label: str
    alias: str


@dataclass(frozen=True, slots=True)
class _JoinStep:
    edge: TypedEdge
    source_sql_alias: str
    destination_sql_alias: str
    destination_table: TableSpec
    user_alias: str


@dataclass(frozen=True, slots=True)
class _ChainEntry:
    sql_alias: str
    user_alias: str
    table: TableSpec


@dataclass(frozen=True, slots=True)
class _WhereClause:
    ref: _RawRef
    op: str
    value: object


@dataclass(frozen=True, slots=True)
class _SelectItem:
    ref: _RawRef
    output_name: str


@dataclass(frozen=True, slots=True)
class _GroupByItem:
    ref: _RawRef
    output_name: str


@dataclass(frozen=True, slots=True)
class _SortClause:
    ref: _RawRef | None
    output_name: str | None
    direction: Literal["asc", "desc"]


@dataclass(frozen=True, slots=True)
class _AggregateClause:
    fn: str
    output_name: str
    ref: _RawRef | None


@dataclass(frozen=True, slots=True)
class _ParsedSpec:
    from_spec: _FromSpec
    where_clauses: list[_WhereClause]
    join_specs: list[_JoinSpec]
    select: list[_SelectItem] | None
    order_by: list[_SortClause]
    limit: int | None
    group_by: list[_GroupByItem]
    aggregates: list[_AggregateClause]


def _require_mapping(raw: object, field: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError(f"{field} must be an object")
    return {str(key): value for key, value in raw.items()}


def _require_known_keys(
    mapping: dict[str, object],
    *,
    field: str,
    allowed: frozenset[str],
) -> None:
    unexpected = set(mapping) - allowed
    if unexpected:
        raise TypeError(
            f"{field} contains unsupported keys: {sorted(unexpected)}"
        )


def _require_non_empty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty string")
    return value.strip()


def _parse_ref(raw: object, *, field: str) -> _RawRef:
    payload = _require_mapping(raw, field)
    _require_known_keys(payload, field=field, allowed=frozenset(("as", "column")))
    return _RawRef(
        alias=_require_non_empty_str(payload.get("as"), f"{field}.as"),
        column=_require_non_empty_str(payload.get("column"), f"{field}.column"),
    )


def _parse_from(raw: object) -> _FromSpec:
    payload = _require_mapping(raw, "from")
    _require_known_keys(payload, field="from", allowed=frozenset(("table", "as")))
    return _FromSpec(
        table=_require_non_empty_str(payload.get("table"), "from.table"),
        alias=_require_non_empty_str(payload.get("as"), "from.as"),
    )


def _parse_join(raw: object) -> list[_JoinSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("'join' must be a list of {from, via_edge, as} objects")
    specs: list[_JoinSpec] = []
    for index, entry in enumerate(raw):
        payload = _require_mapping(entry, f"join[{index}]")
        _require_known_keys(
            payload,
            field=f"join[{index}]",
            allowed=frozenset(("from", "via_edge", "as")),
        )
        source_alias = None
        if payload.get("from") is not None:
            source_alias = _require_non_empty_str(
                payload.get("from"),
                f"join[{index}].from",
            )
        specs.append(
            _JoinSpec(
                source_alias=source_alias,
                edge_label=_require_non_empty_str(
                    payload.get("via_edge"),
                    f"join[{index}].via_edge",
                ),
                alias=_require_non_empty_str(
                    payload.get("as"),
                    f"join[{index}].as",
                ),
            )
        )
    return specs


def _parse_where(raw: object) -> list[_WhereClause]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("'where' must be a list of predicate objects")
    clauses: list[_WhereClause] = []
    for index, entry in enumerate(raw):
        payload = _require_mapping(entry, f"where[{index}]")
        _require_known_keys(
            payload,
            field=f"where[{index}]",
            allowed=frozenset(("ref", "op", "value")),
        )
        op = _require_non_empty_str(payload.get("op"), f"where[{index}].op")
        clauses.append(
            _WhereClause(
                ref=_parse_ref(payload.get("ref"), field=f"where[{index}].ref"),
                op=op,
                value=payload.get("value"),
            )
        )
    return clauses


def _parse_select(raw: object) -> list[_SelectItem] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("'select' must be a list of {ref, as} objects")
    items: list[_SelectItem] = []
    seen_outputs: set[str] = set()
    for index, entry in enumerate(raw):
        payload = _require_mapping(entry, f"select[{index}]")
        _require_known_keys(
            payload,
            field=f"select[{index}]",
            allowed=frozenset(("ref", "as")),
        )
        output_name = _require_non_empty_str(
            payload.get("as"),
            f"select[{index}].as",
        )
        _assert_unique_output_name(output_name, seen_outputs)
        items.append(
            _SelectItem(
                ref=_parse_ref(payload.get("ref"), field=f"select[{index}].ref"),
                output_name=output_name,
            )
        )
    return items


def _parse_group_by(raw: object) -> list[_GroupByItem]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("'group_by' must be a list of {ref, as} objects")
    items: list[_GroupByItem] = []
    seen_outputs: set[str] = set()
    for index, entry in enumerate(raw):
        payload = _require_mapping(entry, f"group_by[{index}]")
        _require_known_keys(
            payload,
            field=f"group_by[{index}]",
            allowed=frozenset(("ref", "as")),
        )
        output_name = _require_non_empty_str(
            payload.get("as"),
            f"group_by[{index}].as",
        )
        _assert_unique_output_name(output_name, seen_outputs)
        items.append(
            _GroupByItem(
                ref=_parse_ref(payload.get("ref"), field=f"group_by[{index}].ref"),
                output_name=output_name,
            )
        )
    return items


def _parse_order_by(raw: object) -> list[_SortClause]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("'order_by' must be a list of order objects")
    clauses: list[_SortClause] = []
    for index, entry in enumerate(raw):
        payload = _require_mapping(entry, f"order_by[{index}]")
        _require_known_keys(
            payload,
            field=f"order_by[{index}]",
            allowed=frozenset(("ref", "output", "direction")),
        )
        direction = _require_non_empty_str(
            payload.get("direction"),
            f"order_by[{index}].direction",
        )
        if direction not in _SORT_DIRECTIONS:
            raise ValueError(
                f"order_by[{index}].direction must be 'asc' or 'desc'"
            )
        has_ref = payload.get("ref") is not None
        has_output = payload.get("output") is not None
        if has_ref == has_output:
            raise TypeError(
                f"order_by[{index}] must include exactly one of 'ref' or 'output'"
            )
        output_name = None
        ref = None
        if has_ref:
            ref = _parse_ref(payload.get("ref"), field=f"order_by[{index}].ref")
        else:
            output_name = _require_non_empty_str(
                payload.get("output"),
                f"order_by[{index}].output",
            )
        clauses.append(
            _SortClause(
                ref=ref,
                output_name=output_name,
                direction="asc" if direction == "asc" else "desc",
            )
        )
    return clauses


def _parse_aggregate(raw: object) -> list[_AggregateClause]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("'aggregate' must be a list of {fn, ref?, as} objects")
    clauses: list[_AggregateClause] = []
    seen_outputs: set[str] = set()
    for index, entry in enumerate(raw):
        payload = _require_mapping(entry, f"aggregate[{index}]")
        _require_known_keys(
            payload,
            field=f"aggregate[{index}]",
            allowed=frozenset(("fn", "ref", "as")),
        )
        fn = _require_non_empty_str(payload.get("fn"), f"aggregate[{index}].fn")
        if fn not in _AGGREGATE_FNS:
            raise ValueError(
                f"aggregate[{index}].fn must be one of "
                f"{sorted(_AGGREGATE_FNS)}; got {fn!r}"
            )
        output_name = _require_non_empty_str(
            payload.get("as"),
            f"aggregate[{index}].as",
        )
        _assert_unique_output_name(output_name, seen_outputs)
        ref: _RawRef | None = None
        if payload.get("ref") is not None:
            ref = _parse_ref(payload.get("ref"), field=f"aggregate[{index}].ref")
        if fn != "count" and ref is None:
            raise TypeError(
                f"aggregate[{index}].ref is required for fn={fn!r}"
            )
        clauses.append(_AggregateClause(fn=fn, output_name=output_name, ref=ref))
    return clauses


def _parse_spec(spec: object) -> _ParsedSpec:
    payload = _require_mapping(spec, "query spec")
    _require_known_keys(
        payload,
        field="query spec",
        allowed=frozenset(
            (
                "from",
                "join",
                "where",
                "select",
                "order_by",
                "limit",
                "group_by",
                "aggregate",
            )
        ),
    )
    from_spec = _parse_from(payload.get("from"))
    limit = (
        _require_int(payload.get("limit"), "limit")
        if payload.get("limit") is not None
        else None
    )
    if limit is not None and limit <= 0:
        raise ValueError("'limit' must be a positive integer")
    select = _parse_select(payload.get("select"))
    aggregates = _parse_aggregate(payload.get("aggregate"))
    group_by = _parse_group_by(payload.get("group_by"))
    if aggregates and select is not None:
        raise ValueError("'select' and 'aggregate' are mutually exclusive")
    if group_by and not aggregates:
        raise ValueError("'group_by' requires at least one aggregate")
    return _ParsedSpec(
        from_spec=from_spec,
        where_clauses=_parse_where(payload.get("where")),
        join_specs=_parse_join(payload.get("join")),
        select=select,
        order_by=_parse_order_by(payload.get("order_by")),
        limit=limit,
        group_by=group_by,
        aggregates=aggregates,
    )


def _resolve_join_chain(
    session: ComposerSession,
    from_spec: _FromSpec,
    join_specs: list[_JoinSpec],
) -> tuple[TableSpec, list[_JoinStep], list[_ChainEntry]]:
    snapshot = session.snapshot
    root = snapshot.table(from_spec.table)
    entries = [_ChainEntry(sql_alias="t0", user_alias=from_spec.alias, table=root)]
    entries_by_alias = {from_spec.alias: entries[0]}
    steps: list[_JoinStep] = []
    current_alias = from_spec.alias
    seen_aliases: set[str] = {from_spec.alias}
    for index, spec in enumerate(join_specs):
        source_alias = spec.source_alias or current_alias
        source_entry = entries_by_alias.get(source_alias)
        if source_entry is None:
            aliases = ", ".join(entry.user_alias for entry in entries)
            raise KeyError(
                f"unknown join source alias {source_alias!r}; available: {aliases}"
            )
        edge = resolve_edge(snapshot, source_entry.table.handle, spec.edge_label)
        destination = snapshot.table(edge.destination_table)
        if spec.alias in seen_aliases:
            raise ValueError(f"duplicate query alias {spec.alias!r}")
        seen_aliases.add(spec.alias)
        destination_sql_alias = f"t{index + 1}"
        steps.append(
            _JoinStep(
                edge=edge,
                source_sql_alias=source_entry.sql_alias,
                destination_sql_alias=destination_sql_alias,
                destination_table=destination,
                user_alias=spec.alias,
            )
        )
        entries.append(
            _ChainEntry(
                sql_alias=destination_sql_alias,
                user_alias=spec.alias,
                table=destination,
            )
        )
        entries_by_alias[spec.alias] = entries[-1]
        current_alias = spec.alias
    return root, steps, entries


def _compile_join_chain(steps: list[_JoinStep]) -> str:
    fragments: list[str] = []
    for step in steps:
        edge = step.edge
        predicates = [
            f"{step.destination_sql_alias}.{quote_ident(destination_column)} "
            f"= {step.source_sql_alias}.{quote_ident(origin_column)}"
            for destination_column, origin_column in zip(
                edge.destination_columns,
                edge.origin_columns,
                strict=True,
            )
        ]
        fragments.append(
            f"JOIN {quote_table(step.destination_table.schema, step.destination_table.name)} "
            f"AS {step.destination_sql_alias} "
            f"ON {' AND '.join(predicates)}"
        )
    return " ".join(fragments)


def _resolve_ref(ref: _RawRef, chain: list[_ChainEntry]) -> _ResolvedRef:
    matches = [entry for entry in chain if entry.user_alias == ref.alias]
    if not matches:
        aliases = ", ".join(entry.user_alias for entry in chain)
        raise KeyError(f"unknown query alias {ref.alias!r}; available: {aliases}")
    entry = matches[0]
    if ref.column not in entry.table.column_names:
        raise KeyError(
            f"column {ref.column!r} not found on alias {ref.alias!r} "
            f"({entry.table.handle})"
        )
    entry.table.exposed_column(ref.column)
    return _ResolvedRef(
        column=ref.column,
        table=entry.table,
        sql_alias=entry.sql_alias,
        user_alias=entry.user_alias,
    )


def _compile_where(
    clauses: list[_WhereClause],
    chain: list[_ChainEntry],
    start_param: int,
) -> tuple[str, tuple[object, ...], int]:
    if not clauses:
        return "", (), start_param
    parts: list[str] = []
    params: list[object] = []
    next_index = start_param
    for clause in clauses:
        if clause.op not in FILTER_OPS:
            raise ValueError(
                f"unsupported op {clause.op!r}; expected one of "
                f"{sorted(FILTER_OPS)}"
            )
        resolved = _resolve_ref(clause.ref, chain)
        column_spec = resolved.table.column(resolved.column)
        qualified = f"{resolved.sql_alias}.{quote_ident(resolved.column)}"
        if clause.op == "is_null":
            parts.append(f"{qualified} IS NULL")
            continue
        if clause.op == "is_not_null":
            parts.append(f"{qualified} IS NOT NULL")
            continue
        value = coerce_scalar(clause.value, column_spec.data_type)
        value = coerce_param(value)
        if clause.op == "in":
            if not isinstance(value, list) or len(value) == 0:
                raise ValueError(
                    f"predicate on {resolved.column!r} with op='in' "
                    "requires a non-empty list value"
                )
            if any(item is None for item in value):
                raise TypeError(
                    f"predicate on {resolved.column!r} with op='in' "
                    "does not accept null list items"
                )
            parts.append(
                f"{qualified} = ANY(${next_index}::"
                f"{array_cast_for(column_spec.data_type)})"
            )
            params.append(list(value))
            next_index += 1
            continue
        if clause.op == "like":
            if value is None:
                raise TypeError(
                    f"predicate on {resolved.column!r} with op='like' "
                    "requires a non-null string pattern"
                )
            if not isinstance(value, str):
                raise TypeError(
                    f"predicate on {resolved.column!r} with op='like' "
                    "requires a string pattern"
                )
            parts.append(f"{qualified} ILIKE ${next_index}")
            params.append(value)
            next_index += 1
            continue
        if value is None:
            raise TypeError(
                f"predicate on {resolved.column!r} with op={clause.op!r} "
                "requires a non-null value"
            )
        sql_op = {
            "eq": "=",
            "neq": "<>",
            "lt": "<",
            "gt": ">",
            "lte": "<=",
            "gte": ">=",
        }[clause.op]
        parts.append(f"{qualified} {sql_op} ${next_index}")
        params.append(value)
        next_index += 1
    return " AND ".join(parts), tuple(params), next_index


def _compile_aggregate_expr(
    aggregate: _AggregateClause,
    chain: list[_ChainEntry],
) -> str:
    if aggregate.fn == "count":
        if aggregate.ref is None:
            return f"COUNT(*) AS {quote_ident(aggregate.output_name)}"
        resolved = _resolve_ref(aggregate.ref, chain)
        return (
            f"COUNT({resolved.sql_alias}.{quote_ident(resolved.column)}) "
            f"AS {quote_ident(aggregate.output_name)}"
        )
    if aggregate.ref is None:
        raise ValueError(f"aggregate fn={aggregate.fn!r} requires a ref")
    resolved = _resolve_ref(aggregate.ref, chain)
    return (
        f"{aggregate.fn.upper()}({resolved.sql_alias}.{quote_ident(resolved.column)}) "
        f"AS {quote_ident(aggregate.output_name)}"
    )


def _compile_sort(
    clause: _SortClause,
    *,
    chain: list[_ChainEntry],
    allowed_outputs: frozenset[str],
) -> str:
    direction = clause.direction.upper()
    if clause.ref is not None:
        resolved = _resolve_ref(clause.ref, chain)
        return f"{resolved.sql_alias}.{quote_ident(resolved.column)} {direction}"
    assert clause.output_name is not None
    if clause.output_name not in allowed_outputs:
        raise KeyError(
            f"unknown output column {clause.output_name!r}; "
            f"available outputs: {sorted(allowed_outputs)}"
        )
    return f"{quote_ident(clause.output_name)} {direction}"


def _column_source_payload(
    *,
    output: str,
    kind: str,
    resolved: _ResolvedRef | None,
    value_exposes_source: bool,
    fn: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "output": output,
        "kind": kind,
        "value_exposes_source": value_exposes_source,
    }
    if fn is not None:
        payload["fn"] = fn
    if resolved is None:
        payload["visibility"] = "derived"
        return payload
    column = resolved.table.column(resolved.column)
    payload.update(
        {
            "table": resolved.table.handle,
            "column": resolved.column,
            "visibility": column.visibility,
            "is_handle": column.is_handle_column,
        }
    )
    return payload


def _referenced_column_payload(
    *,
    usage: str,
    resolved: _ResolvedRef,
    op: str | None = None,
    value: object = None,
    direction: str | None = None,
    output: str | None = None,
) -> dict[str, object]:
    column = resolved.table.column(resolved.column)
    payload: dict[str, object] = {
        "usage": usage,
        "table": resolved.table.handle,
        "column": resolved.column,
        "visibility": column.visibility,
        "is_handle": column.is_handle_column,
    }
    if op is not None:
        payload["op"] = op
        payload["value"] = value
    if direction is not None:
        payload["direction"] = direction
    if output is not None:
        payload["output"] = output
    return payload


def _order_key_outputs(
    clauses: list[_SortClause],
    *,
    chain: list[_ChainEntry],
    column_sources: list[dict[str, object]],
) -> list[str]:
    outputs: list[str] = []
    for clause in clauses:
        if clause.output_name is not None:
            outputs.append(clause.output_name)
            continue
        if clause.ref is None:
            continue
        resolved = _resolve_ref(clause.ref, chain)
        source = next(
            (
                source
                for source in column_sources
                if source.get("table") == resolved.table.handle
                and source.get("column") == resolved.column
                and isinstance(source.get("output"), str)
            ),
            None,
        )
        if source is not None:
            outputs.append(str(source["output"]))
    return outputs


def _ordering_diagnostics(
    rows: list[dict[str, object]],
    *,
    parsed: _ParsedSpec,
    chain: list[_ChainEntry],
    column_sources: list[dict[str, object]],
) -> dict[str, object] | None:
    if parsed.limit is None or len(rows) <= 1:
        return None
    if not parsed.order_by:
        return {
            "missing_order_by_for_limit": True,
            "returned_row_count": len(rows),
            "limit": parsed.limit,
        }
    order_outputs = _order_key_outputs(
        parsed.order_by,
        chain=chain,
        column_sources=column_sources,
    )
    if len(order_outputs) != len(parsed.order_by):
        return None
    signatures = [tuple(repr(row.get(output)) for output in order_outputs) for row in rows]
    duplicate_order_key = len(signatures) != len(set(signatures))
    return {
        "order_by_outputs": order_outputs,
        "duplicate_order_key_in_returned_rows": duplicate_order_key,
        "returned_row_count": len(rows),
        "limit": parsed.limit,
    }


def _assert_unique_output_name(name: str, names: set[str]) -> None:
    if name in names:
        raise ValueError(f"duplicate output column {name!r}")
    names.add(name)


async def query(
    session: ComposerSession,
    *,
    spec: dict[str, object],
) -> dict[str, object]:
    """Execute an authoring query-DSL spec and return rows + column list."""
    parsed = _parse_spec(spec)
    from_spec, steps, chain = _resolve_join_chain(
        session, parsed.from_spec, parsed.join_specs
    )
    target_spec = chain[-1].table
    clause_sql, params, _ = _compile_where(
        parsed.where_clauses,
        chain,
        start_param=1,
    )
    where_sql = f"WHERE {clause_sql}" if clause_sql else ""

    select_fragments: list[str] = []
    output_columns: list[str] = []
    column_sources: list[dict[str, object]] = []
    output_name_set: set[str] = set()
    if parsed.aggregates:
        for group_item in parsed.group_by:
            resolved = _resolve_ref(group_item.ref, chain)
            _assert_unique_output_name(group_item.output_name, output_name_set)
            select_fragments.append(
                f"{resolved.sql_alias}.{quote_ident(resolved.column)} AS "
                f"{quote_ident(group_item.output_name)}"
            )
            output_columns.append(group_item.output_name)
            column_sources.append(
                _column_source_payload(
                    output=group_item.output_name,
                    kind="group_by",
                    resolved=resolved,
                    value_exposes_source=True,
                )
            )
        for aggregate in parsed.aggregates:
            _assert_unique_output_name(aggregate.output_name, output_name_set)
            select_fragments.append(_compile_aggregate_expr(aggregate, chain))
            output_columns.append(aggregate.output_name)
            resolved_ref = (
                _resolve_ref(aggregate.ref, chain)
                if aggregate.ref is not None
                else None
            )
            column_sources.append(
                _column_source_payload(
                    output=aggregate.output_name,
                    kind="aggregate",
                    fn=aggregate.fn,
                    resolved=resolved_ref,
                    value_exposes_source=aggregate.fn != "count",
                )
            )
    elif parsed.select is not None:
        for item in parsed.select:
            resolved = _resolve_ref(item.ref, chain)
            _assert_unique_output_name(item.output_name, output_name_set)
            select_fragments.append(
                f"{resolved.sql_alias}.{quote_ident(resolved.column)} AS "
                f"{quote_ident(item.output_name)}"
            )
            output_columns.append(item.output_name)
            column_sources.append(
                _column_source_payload(
                    output=item.output_name,
                    kind="select",
                    resolved=resolved,
                    value_exposes_source=True,
                )
            )
    else:
        terminal_alias = chain[-1].sql_alias
        for column in target_spec.exposed_columns:
            _assert_unique_output_name(column.name, output_name_set)
            select_fragments.append(
                f"{terminal_alias}.{quote_ident(column.name)} AS "
                f"{quote_ident(column.name)}"
            )
            output_columns.append(column.name)
            column_sources.append(
                {
                    "output": column.name,
                    "kind": "select",
                    "table": target_spec.handle,
                    "column": column.name,
                    "visibility": column.visibility,
                    "is_handle": column.is_handle_column,
                    "value_exposes_source": True,
                }
            )

    referenced_columns: list[dict[str, object]] = []
    for clause in parsed.where_clauses:
        referenced_columns.append(
            _referenced_column_payload(
                usage="where",
                resolved=_resolve_ref(clause.ref, chain),
                op=clause.op,
                value=clause.value,
            )
        )
    for clause in parsed.order_by:
        if clause.ref is not None:
            referenced_columns.append(
                _referenced_column_payload(
                    usage="order_by",
                    resolved=_resolve_ref(clause.ref, chain),
                    direction=clause.direction,
                )
            )
        elif clause.output_name is not None:
            source = next(
                (
                    source
                    for source in column_sources
                    if source.get("output") == clause.output_name
                    and isinstance(source.get("table"), str)
                    and isinstance(source.get("column"), str)
                ),
                None,
            )
            if source is not None:
                referenced_columns.append(
                    {
                        "usage": "order_by",
                        "table": source["table"],
                        "column": source["column"],
                        "visibility": source.get("visibility"),
                        "is_handle": source.get("is_handle"),
                        "direction": clause.direction,
                        "output": clause.output_name,
                    }
                )

    group_by_sql = ""
    if parsed.group_by:
        group_by_fragments: list[str] = []
        for item in parsed.group_by:
            resolved = _resolve_ref(item.ref, chain)
            group_by_fragments.append(
                f"{resolved.sql_alias}.{quote_ident(resolved.column)}"
            )
        group_by_sql = "GROUP BY " + ", ".join(group_by_fragments)

    sort_fragments = [
        _compile_sort(
            clause,
            chain=chain,
            allowed_outputs=frozenset(output_columns),
        )
        for clause in parsed.order_by
    ]
    order_sql = "ORDER BY " + ", ".join(sort_fragments) if sort_fragments else ""
    limit_sql = f"LIMIT {int(parsed.limit)}" if parsed.limit is not None else ""

    join_sql = _compile_join_chain(steps)
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
    result: dict[str, object] = {
        "columns": output_columns,
        "column_sources": column_sources,
        "referenced_columns": referenced_columns,
        "rows": materialized,
        "row_count": len(materialized),
    }
    ordering_diagnostics = _ordering_diagnostics(
        materialized,
        parsed=parsed,
        chain=chain,
        column_sources=column_sources,
    )
    if ordering_diagnostics is not None:
        result["ordering_diagnostics"] = ordering_diagnostics
    return result


__all__ = ["query"]
