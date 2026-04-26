"""Column sensitivity classification."""

from __future__ import annotations

from dataclasses import dataclass

from rl_task_foundry.infra.privacy import Visibility


@dataclass(slots=True)
class ColumnRef:
    schema_name: str
    table_name: str
    column_name: str

    @property
    def qualified_table_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def qualified_column_name(self) -> str:
        return f"{self.qualified_table_name}.{self.column_name}"


@dataclass(slots=True)
class ColumnSensitivity:
    schema_name: str
    table_name: str
    column_name: str
    visibility: Visibility

    @property
    def qualified_column_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}.{self.column_name}"


def resolve_column_visibility(
    column: ColumnRef,
    *,
    default_visibility: Visibility,
    overrides: dict[str, Visibility],
) -> Visibility:
    """Resolve visibility using qualified override precedence.

    Accepted override keys:
    - ``schema.table.column``
    - ``table.column``
    - ``column``
    """

    override_candidates = (
        column.qualified_column_name,
        f"{column.table_name}.{column.column_name}",
        column.column_name,
    )
    for candidate in override_candidates:
        if candidate in overrides:
            return overrides[candidate]

    return default_visibility


def classify_columns(
    columns: list[ColumnRef],
    *,
    default_visibility: Visibility,
    overrides: dict[str, Visibility],
) -> list[ColumnSensitivity]:
    """Attach visibility labels to concrete schema columns."""

    return [
        ColumnSensitivity(
            schema_name=column.schema_name,
            table_name=column.table_name,
            column_name=column.column_name,
            visibility=resolve_column_visibility(
                column,
                default_visibility=default_visibility,
                overrides=overrides,
            ),
        )
        for column in columns
    ]
