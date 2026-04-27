"""Immutable schema snapshot used by atomic/composer toolsets.

The snapshot is decoupled from the mutable `schema.graph.SchemaGraph` so that
toolsets can be built from frozen metadata (and later from a bundle-level
`schema_snapshot.json` without re-introspecting the database).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from string import ascii_letters, digits

from rl_task_foundry.infra.visibility import (
    VISIBILITY_BLOCKED,
    VISIBILITY_USER_VISIBLE,
    Visibility,
    is_visibility,
)
from rl_task_foundry.schema.graph import SchemaGraph


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    visibility: Visibility

    @property
    def is_handle_column(self) -> bool:
        return self.is_foreign_key or self.is_primary_key

    @property
    def is_exposed(self) -> bool:
        return self.visibility != VISIBILITY_BLOCKED or self.is_handle_column


@dataclass(frozen=True, slots=True)
class TableSpec:
    schema: str
    name: str
    handle: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...]

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def column(self, column_name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == column_name:
                return column
        raise KeyError(f"{self.qualified_name}.{column_name}")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def exposed_columns(self) -> tuple[ColumnSpec, ...]:
        return tuple(column for column in self.columns if column.is_exposed)

    @property
    def exposed_column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.exposed_columns)

    def exposed_column(self, column_name: str) -> ColumnSpec:
        column = self.column(column_name)
        if not column.is_exposed:
            raise KeyError(f"{self.qualified_name}.{column_name}")
        return column


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """Foreign key edge.

    `source_table` and `target_table` are canonical table handles. They
    are the bare table name when globally unique in the snapshot, and
    `schema.table` when disambiguation is needed.
    """

    source_table: str
    source_columns: tuple[str, ...]
    target_table: str
    target_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_columns or not self.target_columns:
            raise ValueError("foreign-key edges require at least one column")
        if len(self.source_columns) != len(self.target_columns):
            raise ValueError(
                "foreign-key source_columns and target_columns must have "
                "the same length"
            )

    @property
    def source_column(self) -> str:
        if len(self.source_columns) != 1:
            raise ValueError("source_column is only available for single-column FKs")
        return self.source_columns[0]

    @property
    def target_column(self) -> str:
        if len(self.target_columns) != 1:
            raise ValueError("target_column is only available for single-column FKs")
        return self.target_columns[0]

    @property
    def forward_label(self) -> str:
        if len(self.source_columns) == 1:
            return (
                f"{_edge_label_part(self.source_table)}."
                f"{_edge_label_part(self.source_column)}"
                f"->{_edge_label_part(self.target_table)}"
            )
        source = _edge_column_label(self.source_columns)
        target = _edge_column_label(self.target_columns)
        return (
            f"{_edge_label_part(self.source_table)}.{source}"
            f"->{_edge_label_part(self.target_table)}.{target}"
        )

    @property
    def reverse_label(self) -> str:
        if len(self.source_columns) == 1:
            return (
                f"{_edge_label_part(self.target_table)}"
                f"<-{_edge_label_part(self.source_table)}."
                f"{_edge_label_part(self.source_column)}"
            )
        source = _edge_column_label(self.source_columns)
        target = _edge_column_label(self.target_columns)
        return (
            f"{_edge_label_part(self.target_table)}.{target}"
            f"<-{_edge_label_part(self.source_table)}.{source}"
        )


@dataclass(frozen=True, slots=True)
class SchemaSnapshot:
    tables: tuple[TableSpec, ...]
    edges: tuple[EdgeSpec, ...]

    def table(self, table_name: str) -> TableSpec:
        for table in self.tables:
            if table.handle == table_name:
                return table
        qualified_matches = [
            table for table in self.tables if table.qualified_name == table_name
        ]
        if len(qualified_matches) == 1:
            return qualified_matches[0]
        if len(qualified_matches) > 1:
            raise KeyError(
                f"{table_name} matches multiple qualified table names; "
                "use a schema snapshot table handle"
            )
        name_matches = [table for table in self.tables if table.name == table_name]
        if len(name_matches) == 1:
            return name_matches[0]
        raise KeyError(table_name)

    def table_handle(self, table_name: str) -> str:
        return self.table(table_name).handle

    def table_names(self) -> tuple[str, ...]:
        return tuple(table.handle for table in self.tables)

    def edges_from(self, table_name: str) -> tuple[EdgeSpec, ...]:
        table_handle = self.table_handle(table_name)
        return tuple(
            edge for edge in self.edges if edge.source_table == table_handle
        )

    def edges_to(self, table_name: str) -> tuple[EdgeSpec, ...]:
        table_handle = self.table_handle(table_name)
        return tuple(
            edge for edge in self.edges if edge.target_table == table_handle
        )


def snapshot_from_graph(graph: SchemaGraph) -> SchemaSnapshot:
    tables: list[TableSpec] = []
    name_counts = Counter(table.table_name for table in graph.tables)
    qualified_names = {table.qualified_name for table in graph.tables}
    table_handles: dict[tuple[str, str], str] = {}
    for table in graph.tables:
        handle = _canonical_table_handle(
            schema=table.schema_name,
            name=table.table_name,
            table_name_counts=name_counts,
            qualified_names=qualified_names,
        )
        table_handles[(table.schema_name, table.table_name)] = handle
        columns = tuple(
            ColumnSpec(
                name=column.column_name,
                data_type=column.data_type,
                is_nullable=column.is_nullable,
                is_primary_key=column.is_primary_key,
                is_foreign_key=column.is_foreign_key,
                visibility=column.visibility,
            )
            for column in table.columns
        )
        tables.append(
            TableSpec(
                schema=table.schema_name,
                name=table.table_name,
                handle=handle,
                columns=columns,
                primary_key=tuple(table.primary_key),
            )
        )
    _ensure_unique_table_handles(tables)
    edges: list[EdgeSpec] = []
    for edge in graph.edges:
        if not edge.source_columns or not edge.target_columns:
            raise ValueError(
                f"foreign key {edge.constraint_name!r} has no column pairs"
            )
        if len(edge.source_columns) != len(edge.target_columns):
            raise ValueError(
                f"foreign key {edge.constraint_name!r} has mismatched "
                "source/target column counts"
            )
        edges.append(
            EdgeSpec(
                source_table=table_handles[(edge.source_schema, edge.source_table)],
                source_columns=tuple(edge.source_columns),
                target_table=table_handles[(edge.target_schema, edge.target_table)],
                target_columns=tuple(edge.target_columns),
            )
        )
    edge_specs = _dedupe_edges(edges)
    _ensure_unique_edge_labels(edge_specs)
    return SchemaSnapshot(tables=tuple(tables), edges=edge_specs)


def snapshot_to_dict(snapshot: SchemaSnapshot) -> dict[str, object]:
    """Serialize a `SchemaSnapshot` to a JSON-ready dict.

    Round-trips with `snapshot_from_dict` — used by the bundle export
    flow so env servers can load the snapshot and re-instantiate the
    atomic calculus tools without re-introspecting the database.
    """
    tables: list[dict[str, object]] = []
    for table in snapshot.tables:
        tables.append(
            {
                "schema": table.schema,
                "name": table.name,
                "handle": table.handle,
                "primary_key": list(table.primary_key),
                "columns": [
                    {
                        "name": column.name,
                        "data_type": column.data_type,
                        "is_nullable": column.is_nullable,
                        "is_primary_key": column.is_primary_key,
                        "is_foreign_key": column.is_foreign_key,
                        "visibility": column.visibility,
                    }
                    for column in table.columns
                ],
            }
        )
    edges: list[dict[str, object]] = [
        {
            "source_table": edge.source_table,
            "source_columns": list(edge.source_columns),
            "target_table": edge.target_table,
            "target_columns": list(edge.target_columns),
        }
        for edge in snapshot.edges
    ]
    return {"tables": tables, "edges": edges}


def _require_str(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"schema snapshot payload missing string field {key!r}"
        )
    return value


def _require_bool(mapping: dict[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(
            f"schema snapshot payload missing bool field {key!r}"
        )
    return value


def snapshot_from_dict(payload: dict[str, object]) -> SchemaSnapshot:
    """Rebuild a `SchemaSnapshot` from the JSON shape written by
    `snapshot_to_dict`. Raises `ValueError` on structural mismatch.
    """
    tables_raw = payload.get("tables")
    edges_raw = payload.get("edges")
    if not isinstance(tables_raw, list):
        raise ValueError("schema snapshot payload missing 'tables' list")
    if not isinstance(edges_raw, list):
        raise ValueError("schema snapshot payload missing 'edges' list")
    table_specs: list[TableSpec] = []
    table_entries: list[dict[str, object]] = []
    table_name_counts: Counter[str] = Counter()
    qualified_names: set[str] = set()
    for index, table_entry in enumerate(tables_raw):
        if not isinstance(table_entry, dict):
            raise ValueError(f"tables[{index}] must be a mapping")
        typed_entry: dict[str, object] = {
            str(key): value for key, value in table_entry.items()
        }
        table_entries.append(typed_entry)
        schema = _require_str(typed_entry, "schema")
        name = _require_str(typed_entry, "name")
        table_name_counts[name] += 1
        qualified_names.add(f"{schema}.{name}")
    for index, typed_entry in enumerate(table_entries):
        columns_raw = typed_entry.get("columns")
        if not isinstance(columns_raw, list):
            raise ValueError(f"tables[{index}].columns must be a list")
        column_specs: list[ColumnSpec] = []
        for column_index, column_entry in enumerate(columns_raw):
            if not isinstance(column_entry, dict):
                raise ValueError(
                    f"tables[{index}].columns[{column_index}] must be a mapping"
                )
            column_typed: dict[str, object] = {
                str(key): value for key, value in column_entry.items()
            }
            column_specs.append(
                ColumnSpec(
                    name=_require_str(column_typed, "name"),
                    data_type=_require_str(column_typed, "data_type"),
                    is_nullable=_require_bool(column_typed, "is_nullable"),
                    is_primary_key=_require_bool(column_typed, "is_primary_key"),
                    is_foreign_key=_require_bool(column_typed, "is_foreign_key"),
                    visibility=_visibility_or_default(column_typed),
                )
            )
        pk_raw = typed_entry.get("primary_key", [])
        if not isinstance(pk_raw, list):
            raise ValueError(
                f"tables[{index}].primary_key must be a list"
            )
        pk_columns = tuple(str(column) for column in pk_raw)
        table_specs.append(
            TableSpec(
                schema=_require_str(typed_entry, "schema"),
                name=_require_str(typed_entry, "name"),
                handle=_table_handle_from_payload(
                    typed_entry,
                    table_name_counts=table_name_counts,
                    qualified_names=qualified_names,
                ),
                columns=tuple(column_specs),
                primary_key=pk_columns,
        )
    )
    _ensure_unique_table_handles(table_specs)
    table_handles = {table.handle for table in table_specs}
    name_to_handles: dict[str, list[str]] = {}
    qualified_to_handle: dict[str, str] = {}
    for table in table_specs:
        name_to_handles.setdefault(table.name, []).append(table.handle)
        qualified_to_handle[table.qualified_name] = table.handle
    edge_specs: list[EdgeSpec] = []
    for index, edge_entry in enumerate(edges_raw):
        if not isinstance(edge_entry, dict):
            raise ValueError(f"edges[{index}] must be a mapping")
        edge_typed: dict[str, object] = {
            str(key): value for key, value in edge_entry.items()
        }
        edge_specs.append(
            EdgeSpec(
                source_table=_resolve_edge_table_handle(
                    _require_str(edge_typed, "source_table"),
                    table_handles=table_handles,
                    name_to_handles=name_to_handles,
                    qualified_to_handle=qualified_to_handle,
                ),
                source_columns=_edge_columns_from_payload(
                    edge_typed,
                    plural_key="source_columns",
                    singular_key="source_column",
                ),
                target_table=_resolve_edge_table_handle(
                    _require_str(edge_typed, "target_table"),
                    table_handles=table_handles,
                    name_to_handles=name_to_handles,
                    qualified_to_handle=qualified_to_handle,
                ),
                target_columns=_edge_columns_from_payload(
                    edge_typed,
                    plural_key="target_columns",
                    singular_key="target_column",
                ),
            )
        )
    edge_specs_deduped = _dedupe_edges(edge_specs)
    _ensure_unique_edge_labels(edge_specs_deduped)
    return SchemaSnapshot(tables=tuple(table_specs), edges=edge_specs_deduped)


def _visibility_or_default(mapping: dict[str, object]) -> Visibility:
    value = mapping.get("visibility", VISIBILITY_USER_VISIBLE)
    if is_visibility(value):
        return value
    raise ValueError(
        "schema snapshot column visibility must be one of "
        "'blocked', 'internal', or 'user_visible'"
    )


def _edge_column_label(columns: tuple[str, ...]) -> str:
    if len(columns) == 1:
        return _edge_label_part(columns[0])
    return "(" + ",".join(_edge_label_part(column) for column in columns) + ")"


_EDGE_LABEL_SAFE_CHARS = frozenset(ascii_letters + digits + "_")


def _edge_label_part(value: str) -> str:
    """Encode one relation-label component without label delimiters.

    Relation labels are opaque IDs copied from tool output into subsequent
    calls. Keeping simple SQL-style identifiers readable is useful, but arbitrary
    PostgreSQL identifiers may contain `.`, `,`, `->`, `<-`, or parentheses.
    Percent-encoding every non-alphanumeric/underscore byte prevents those
    identifiers from colliding with our label separators.
    """

    parts: list[str] = []
    for char in value:
        if char in _EDGE_LABEL_SAFE_CHARS:
            parts.append(char)
            continue
        parts.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
    return "".join(parts)


def _dedupe_edges(edges: list[EdgeSpec]) -> tuple[EdgeSpec, ...]:
    seen: set[EdgeSpec] = set()
    deduped: list[EdgeSpec] = []
    for edge in edges:
        if edge in seen:
            continue
        seen.add(edge)
        deduped.append(edge)
    return tuple(deduped)


def _ensure_unique_edge_labels(edges: tuple[EdgeSpec, ...]) -> None:
    labels_by_origin: dict[tuple[str, str], EdgeSpec] = {}
    for edge in edges:
        for origin, label in (
            (edge.source_table, edge.forward_label),
            (edge.target_table, edge.reverse_label),
        ):
            key = (origin, label)
            previous = labels_by_origin.get(key)
            if previous is not None and previous != edge:
                raise ValueError(
                    "schema snapshot relation labels must be unique per "
                    f"origin table: {origin!r} has duplicate label {label!r}"
                )
            labels_by_origin[key] = edge


def _edge_columns_from_payload(
    mapping: dict[str, object],
    *,
    plural_key: str,
    singular_key: str,
) -> tuple[str, ...]:
    value = mapping.get(plural_key)
    if isinstance(value, list):
        columns = tuple(str(column) for column in value)
        if columns and all(columns):
            return columns
    if value is not None:
        raise ValueError(
            f"schema snapshot edge field {plural_key!r} must be a non-empty list"
        )
    return (_require_str(mapping, singular_key),)


def _ensure_unique_table_handles(
    tables: list[TableSpec] | tuple[TableSpec, ...],
) -> None:
    handle_counts = Counter(table.handle for table in tables)
    duplicates = sorted(
        handle for handle, count in handle_counts.items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"schema snapshot table handles must be unique: {duplicates}"
        )


def _canonical_table_handle(
    *,
    schema: str,
    name: str,
    table_name_counts: Counter[str],
    qualified_names: set[str],
) -> str:
    can_use_bare_name = (
        table_name_counts[name] == 1
        and "." not in name
        and name not in qualified_names
    )
    if can_use_bare_name:
        return name
    return f"{schema}.{name}"


def _table_handle_from_payload(
    mapping: dict[str, object],
    *,
    table_name_counts: Counter[str],
    qualified_names: set[str],
) -> str:
    handle = mapping.get("handle")
    if isinstance(handle, str) and handle.strip():
        return handle
    schema = _require_str(mapping, "schema")
    name = _require_str(mapping, "name")
    return _canonical_table_handle(
        schema=schema,
        name=name,
        table_name_counts=table_name_counts,
        qualified_names=qualified_names,
    )


def _resolve_edge_table_handle(
    value: str,
    *,
    table_handles: set[str],
    name_to_handles: dict[str, list[str]],
    qualified_to_handle: dict[str, str],
) -> str:
    if value in table_handles:
        return value
    if value in qualified_to_handle:
        return qualified_to_handle[value]
    handles = name_to_handles.get(value, [])
    if len(handles) == 1:
        return handles[0]
    raise ValueError(f"ambiguous or unknown table handle {value!r}")
