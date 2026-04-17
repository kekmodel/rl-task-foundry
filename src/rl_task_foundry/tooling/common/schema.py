"""Immutable schema snapshot used by atomic/composer toolsets.

The snapshot is decoupled from the mutable `schema.graph.SchemaGraph` so that
toolsets can be built from frozen metadata (and later from a bundle-level
`schema_snapshot.json` without re-introspecting the database).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rl_task_foundry.schema.graph import SchemaGraph


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool


@dataclass(frozen=True, slots=True)
class TableSpec:
    schema: str
    name: str
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


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """Single-column foreign key edge.

    Composite FKs are not represented; they will be handled in a later
    extension and are rare in the current schemas (sakila, pagila).
    """

    source_table: str
    source_column: str
    target_table: str
    target_column: str

    @property
    def forward_label(self) -> str:
        return f"{self.source_table}.{self.source_column}->{self.target_table}"


@dataclass(frozen=True, slots=True)
class SchemaSnapshot:
    tables: tuple[TableSpec, ...]
    edges: tuple[EdgeSpec, ...]

    def table(self, table_name: str) -> TableSpec:
        for table in self.tables:
            if table.name == table_name:
                return table
        raise KeyError(table_name)

    def table_names(self) -> tuple[str, ...]:
        return tuple(table.name for table in self.tables)

    def edges_from(self, table_name: str) -> tuple[EdgeSpec, ...]:
        return tuple(edge for edge in self.edges if edge.source_table == table_name)

    def edges_to(self, table_name: str) -> tuple[EdgeSpec, ...]:
        return tuple(edge for edge in self.edges if edge.target_table == table_name)


def snapshot_from_graph(graph: SchemaGraph) -> SchemaSnapshot:
    tables: list[TableSpec] = []
    for table in graph.tables:
        columns = tuple(
            ColumnSpec(
                name=column.column_name,
                data_type=column.data_type,
                is_nullable=column.is_nullable,
                is_primary_key=column.is_primary_key,
                is_foreign_key=column.is_foreign_key,
            )
            for column in table.columns
        )
        tables.append(
            TableSpec(
                schema=table.schema_name,
                name=table.table_name,
                columns=columns,
                primary_key=tuple(table.primary_key),
            )
        )
    edges = tuple(
        EdgeSpec(
            source_table=edge.source_table,
            source_column=edge.source_columns[0],
            target_table=edge.target_table,
            target_column=edge.target_columns[0],
        )
        for edge in graph.edges
        if len(edge.source_columns) == 1 and len(edge.target_columns) == 1
    )
    return SchemaSnapshot(tables=tuple(tables), edges=edges)


def _iter_table_names(snapshot: SchemaSnapshot) -> Iterable[str]:
    for table in snapshot.tables:
        yield table.name


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
                "primary_key": list(table.primary_key),
                "columns": [
                    {
                        "name": column.name,
                        "data_type": column.data_type,
                        "is_nullable": column.is_nullable,
                        "is_primary_key": column.is_primary_key,
                        "is_foreign_key": column.is_foreign_key,
                    }
                    for column in table.columns
                ],
            }
        )
    edges: list[dict[str, object]] = [
        {
            "source_table": edge.source_table,
            "source_column": edge.source_column,
            "target_table": edge.target_table,
            "target_column": edge.target_column,
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
    for index, table_entry in enumerate(tables_raw):
        if not isinstance(table_entry, dict):
            raise ValueError(f"tables[{index}] must be a mapping")
        typed_entry: dict[str, object] = {
            str(key): value for key, value in table_entry.items()
        }
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
                columns=tuple(column_specs),
                primary_key=pk_columns,
            )
        )
    edge_specs: list[EdgeSpec] = []
    for index, edge_entry in enumerate(edges_raw):
        if not isinstance(edge_entry, dict):
            raise ValueError(f"edges[{index}] must be a mapping")
        edge_typed: dict[str, object] = {
            str(key): value for key, value in edge_entry.items()
        }
        edge_specs.append(
            EdgeSpec(
                source_table=_require_str(edge_typed, "source_table"),
                source_column=_require_str(edge_typed, "source_column"),
                target_table=_require_str(edge_typed, "target_table"),
                target_column=_require_str(edge_typed, "target_column"),
            )
        )
    return SchemaSnapshot(tables=tuple(table_specs), edges=tuple(edge_specs))
