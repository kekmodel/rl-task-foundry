"""Schema graph and table metadata models."""

from __future__ import annotations

from dataclasses import dataclass, field

from rl_task_foundry.infra.privacy import Visibility


@dataclass(slots=True)
class ColumnProfile:
    schema_name: str
    table_name: str
    column_name: str
    data_type: str
    ordinal_position: int
    is_nullable: bool
    visibility: Visibility
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}.{self.column_name}"


@dataclass(slots=True)
class TableProfile:
    schema_name: str
    table_name: str
    columns: list[ColumnProfile] = field(default_factory=list)
    primary_key: tuple[str, ...] = ()
    unique_constraints: list[tuple[str, ...]] = field(default_factory=list)
    row_estimate: int | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def nullable_columns(self) -> list[str]:
        return [column.column_name for column in self.columns if column.is_nullable]

    @property
    def unique_columns(self) -> list[str]:
        return [column.column_name for column in self.columns if column.is_unique]

    def get_column(self, column_name: str) -> ColumnProfile:
        for column in self.columns:
            if column.column_name == column_name:
                return column
        raise KeyError(f"{self.qualified_name}.{column_name}")


@dataclass(slots=True)
class ForeignKeyEdge:
    constraint_name: str
    source_schema: str
    source_table: str
    source_columns: tuple[str, ...]
    target_schema: str
    target_table: str
    target_columns: tuple[str, ...]
    source_is_unique: bool = False
    fanout_estimate: float | None = None

    @property
    def source_qualified_name(self) -> str:
        return f"{self.source_schema}.{self.source_table}"

    @property
    def target_qualified_name(self) -> str:
        return f"{self.target_schema}.{self.target_table}"


@dataclass(slots=True)
class SchemaGraph:
    tables: list[TableProfile] = field(default_factory=list)
    edges: list[ForeignKeyEdge] = field(default_factory=list)

    def get_table(self, table_name: str, *, schema_name: str | None = None) -> TableProfile:
        for table in self.tables:
            if table.table_name != table_name:
                continue
            if schema_name is not None and table.schema_name != schema_name:
                continue
            return table
        if schema_name is None:
            raise KeyError(table_name)
        raise KeyError(f"{schema_name}.{table_name}")

    def table_names(self) -> list[str]:
        return [table.table_name for table in self.tables]

    def qualified_table_names(self) -> list[str]:
        return [table.qualified_name for table in self.tables]

    def iter_columns(self) -> list[ColumnProfile]:
        return [column for table in self.tables for column in table.columns]

    def edges_from(self, table_name: str, *, schema_name: str | None = None) -> list[ForeignKeyEdge]:
        return [
            edge
            for edge in self.edges
            if edge.source_table == table_name
            and (schema_name is None or edge.source_schema == schema_name)
        ]

    def edges_to(self, table_name: str, *, schema_name: str | None = None) -> list[ForeignKeyEdge]:
        return [
            edge
            for edge in self.edges
            if edge.target_table == table_name
            and (schema_name is None or edge.target_schema == schema_name)
        ]
