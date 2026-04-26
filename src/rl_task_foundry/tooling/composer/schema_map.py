"""schema_map primitive — inspect a schema graph.

Returns a JSON-ready slice of the schema centered at `root_table`
(or the whole schema when `root_table is None`), with hub/bridge tags
derived from FK incidence. The implementation is pure: it reads only
the `SchemaSnapshot` and does not touch the database.

hub: tables in the top third of the (in-degree + out-degree) ranking,
     i.e. heavy FK participants. Useful for agents looking for
     "everywhere-referenced" entities.

bridge: tables with two or more outgoing FK edges. Many m:n junction tables
     have this shape and often serve as stepping stones between otherwise
     disconnected subgraphs.
"""

from __future__ import annotations

from collections import deque

from rl_task_foundry.tooling.common.edges import available_edges
from rl_task_foundry.tooling.common.schema import (
    EdgeSpec,
    SchemaSnapshot,
    TableSpec,
)


def _describe_table(table: TableSpec) -> dict[str, object]:
    return {
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
            for column in table.exposed_columns
        ],
    }


def _describe_edge(edge: EdgeSpec) -> dict[str, object]:
    return {
        "source_table": edge.source_table,
        "source_columns": list(edge.source_columns),
        "target_table": edge.target_table,
        "target_columns": list(edge.target_columns),
        "forward_label": edge.forward_label,
    }


def _classify_hubs(
    snapshot: SchemaSnapshot, table_names: list[str]
) -> list[str]:
    degree: dict[str, int] = {}
    for name in table_names:
        degree[name] = len(snapshot.edges_from(name)) + len(
            snapshot.edges_to(name)
        )
    if not degree:
        return []
    ranked = sorted(degree.items(), key=lambda pair: (-pair[1], pair[0]))
    # Top-third by degree, minimum 1 entry when at least one has FK incidence.
    cutoff = max(1, len(ranked) // 3)
    hubs = [name for name, count in ranked[:cutoff] if count > 0]
    return hubs


def _classify_bridges(
    snapshot: SchemaSnapshot, table_names: list[str]
) -> list[str]:
    bridges: list[str] = []
    for name in table_names:
        outgoing = snapshot.edges_from(name)
        if len(outgoing) >= 2:
            bridges.append(name)
    return bridges


def _bfs_tables(
    snapshot: SchemaSnapshot, root: str, depth: int
) -> list[str]:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    root_handle = snapshot.table(root).handle
    visited: set[str] = {root_handle}
    order: list[str] = [root_handle]
    if depth == 0:
        return order
    frontier: deque[tuple[str, int]] = deque([(root_handle, 0)])
    while frontier:
        current, current_depth = frontier.popleft()
        if current_depth >= depth:
            continue
        neighbors: list[str] = []
        for edge in available_edges(snapshot, current):
            neighbors.append(edge.destination_table)
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            order.append(neighbor)
            frontier.append((neighbor, current_depth + 1))
    return order


def _edges_within(
    snapshot: SchemaSnapshot, table_names: set[str]
) -> list[EdgeSpec]:
    return [
        edge
        for edge in snapshot.edges
        if edge.source_table in table_names
        and edge.target_table in table_names
    ]


def _typed_edge_labels(
    snapshot: SchemaSnapshot, origin_table: str
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for edge in available_edges(snapshot, origin_table):
        entries.append(
            {
                "label": edge.label,
                "direction": edge.direction.value,
                "destination_table": edge.destination_table,
            }
        )
    return entries


def schema_map(
    snapshot: SchemaSnapshot,
    *,
    root_table: str | None = None,
    depth: int = 2,
) -> dict[str, object]:
    """Return a JSON-ready slice of the schema graph.

    `root_table=None` returns the whole schema (depth is ignored).
    Otherwise a depth-limited BFS over FK edges selects the neighborhood
    around `root_table`. hub/bridge tags are computed over the returned
    slice so they highlight local structure.
    """
    if root_table is None:
        canonical_root = None
        included_names = [table.handle for table in snapshot.tables]
    else:
        canonical_root = snapshot.table(root_table).handle
        included_names = _bfs_tables(snapshot, canonical_root, depth)
    included_set = set(included_names)
    tables_payload = [
        _describe_table(snapshot.table(name)) for name in included_names
    ]
    edge_specs = _edges_within(snapshot, included_set)
    edges_payload = [_describe_edge(edge) for edge in edge_specs]
    typed_edges = {
        name: _typed_edge_labels(snapshot, name) for name in included_names
    }
    return {
        "root_table": canonical_root,
        "depth": depth,
        "tables": tables_payload,
        "edges": edges_payload,
        "typed_edges": typed_edges,
        "hub_tables": _classify_hubs(snapshot, included_names),
        "bridge_tables": _classify_bridges(snapshot, included_names),
    }


__all__ = ["schema_map"]
