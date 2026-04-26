"""Typed foreign-key edges exposed to the atomic calculus.

Each edge has a direction from an origin table to a destination table.
`rows_via(cursor, edge)` projects a cursor over `edge.origin_table` into a
cursor over `edge.destination_table` by joining through the FK. The
implementation supports both single-column and composite FKs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rl_task_foundry.tooling.common.schema import EdgeSpec, SchemaSnapshot


class EdgeDirection(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


@dataclass(frozen=True, slots=True)
class TypedEdge:
    """Directed edge used by the atomic calculus.

    For a FK `source_table.source_columns -> target_table.target_columns`:
    - FORWARD direction moves cursors from source rows to target rows
      (many-to-one in the common case).
    - REVERSE direction moves cursors from target rows to source rows
      (one-to-many).

    The atomic resource API treats the traversal result as a set of unique
    destination records by primary key.
    """

    spec: EdgeSpec
    direction: EdgeDirection

    @property
    def origin_table(self) -> str:
        if self.direction is EdgeDirection.FORWARD:
            return self.spec.source_table
        return self.spec.target_table

    @property
    def destination_table(self) -> str:
        if self.direction is EdgeDirection.FORWARD:
            return self.spec.target_table
        return self.spec.source_table

    @property
    def origin_column(self) -> str:
        if len(self.origin_columns) != 1:
            raise ValueError("origin_column is only available for single-column FKs")
        return self.origin_columns[0]

    @property
    def destination_column(self) -> str:
        if len(self.destination_columns) != 1:
            raise ValueError(
                "destination_column is only available for single-column FKs"
            )
        return self.destination_columns[0]

    @property
    def origin_columns(self) -> tuple[str, ...]:
        if self.direction is EdgeDirection.FORWARD:
            return self.spec.source_columns
        return self.spec.target_columns

    @property
    def destination_columns(self) -> tuple[str, ...]:
        if self.direction is EdgeDirection.FORWARD:
            return self.spec.target_columns
        return self.spec.source_columns

    @property
    def label(self) -> str:
        if self.direction is EdgeDirection.FORWARD:
            return self.spec.forward_label
        return self.spec.reverse_label


def available_edges(
    snapshot: SchemaSnapshot, origin_table: str
) -> tuple[TypedEdge, ...]:
    """All typed edges whose origin_table matches `origin_table`."""
    edges: list[TypedEdge] = []
    for spec in snapshot.edges_from(origin_table):
        edges.append(TypedEdge(spec=spec, direction=EdgeDirection.FORWARD))
    for spec in snapshot.edges_to(origin_table):
        edges.append(TypedEdge(spec=spec, direction=EdgeDirection.REVERSE))
    return tuple(edges)


def resolve_edge(
    snapshot: SchemaSnapshot, origin_table: str, label: str
) -> TypedEdge:
    """Resolve an edge label to a TypedEdge, validating the origin.

    Raises KeyError if no edge with that label originates at `origin_table`.
    """
    for edge in available_edges(snapshot, origin_table):
        if edge.label == label:
            return edge
    raise KeyError(
        f"no edge labelled {label!r} originates at table {origin_table!r}"
    )
