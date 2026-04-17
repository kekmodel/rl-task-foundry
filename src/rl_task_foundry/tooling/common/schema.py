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
