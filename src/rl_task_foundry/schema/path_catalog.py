"""Path catalog and difficulty features."""

from __future__ import annotations

from dataclasses import dataclass, field

from rl_task_foundry.schema.graph import ForeignKeyEdge, SchemaGraph

DifficultyValue = int | float | str | bool


@dataclass(slots=True)
class DifficultyFeatures:
    required_hops: int
    fanout_max: float
    fanout_product: float
    cardinality_estimate: float | None
    has_unique_join: bool
    has_nullable_hop: bool
    shortcut_count: int

    def as_dict(self) -> dict[str, DifficultyValue]:
        return {
            "required_hops": self.required_hops,
            "fanout_max": self.fanout_max,
            "fanout_product": self.fanout_product,
            "cardinality_estimate": self.cardinality_estimate or 0.0,
            "has_unique_join": self.has_unique_join,
            "has_nullable_hop": self.has_nullable_hop,
            "shortcut_count": self.shortcut_count,
        }


@dataclass(slots=True)
class PathSpec:
    path_id: str
    root_table: str
    tables: list[str]
    edges: list[ForeignKeyEdge]
    hop_count: int
    shortcut_candidates: list[str] = field(default_factory=list)
    difficulty_features: dict[str, DifficultyValue] = field(default_factory=dict)


@dataclass(slots=True)
class PathCatalog:
    paths: list[PathSpec] = field(default_factory=list)

    def get(self, path_id: str) -> PathSpec:
        for path in self.paths:
            if path.path_id == path_id:
                return path
        raise KeyError(path_id)

    def for_root(self, table_name: str) -> list[PathSpec]:
        return [path for path in self.paths if path.root_table == table_name]


def build_path_catalog(graph: SchemaGraph, *, max_hops: int) -> PathCatalog:
    """Enumerate simple FK paths for every table in the schema graph."""

    collected: dict[str, PathSpec] = {}
    for table in graph.tables:
        root_key = (table.schema_name, table.table_name)
        _walk_paths(
            graph,
            root_key=root_key,
            current_key=root_key,
            visited={root_key},
            traversed_edges=[],
            collected=collected,
            max_hops=max_hops,
        )
    return PathCatalog(paths=sorted(collected.values(), key=lambda path: path.path_id))


def _walk_paths(
    graph: SchemaGraph,
    *,
    root_key: tuple[str, str],
    current_key: tuple[str, str],
    visited: set[tuple[str, str]],
    traversed_edges: list[ForeignKeyEdge],
    collected: dict[str, PathSpec],
    max_hops: int,
) -> None:
    if traversed_edges:
        path = _path_spec_for_edges(graph, root_key=root_key, edges=traversed_edges)
        collected[path.path_id] = path
    if len(traversed_edges) >= max_hops:
        return

    for edge in graph.edges_from(current_key[1], schema_name=current_key[0]):
        next_key = (edge.target_schema, edge.target_table)
        if next_key in visited:
            continue
        _walk_paths(
            graph,
            root_key=root_key,
            current_key=next_key,
            visited=visited | {next_key},
            traversed_edges=[*traversed_edges, edge],
            collected=collected,
            max_hops=max_hops,
        )


def _path_spec_for_edges(
    graph: SchemaGraph,
    *,
    root_key: tuple[str, str],
    edges: list[ForeignKeyEdge],
) -> PathSpec:
    root_table = root_key[1]
    tables = [root_table, *[edge.target_table for edge in edges]]
    path_id = ".".join(tables)
    shortcuts = _find_shortcut_candidates(graph, tables=tables, path_id=path_id)
    features = _build_difficulty_features(graph, edges=edges, shortcut_candidates=shortcuts)
    return PathSpec(
        path_id=path_id,
        root_table=root_table,
        tables=tables,
        edges=list(edges),
        hop_count=len(edges),
        shortcut_candidates=shortcuts,
        difficulty_features=features.as_dict(),
    )


def _find_shortcut_candidates(graph: SchemaGraph, *, tables: list[str], path_id: str) -> list[str]:
    shortcuts: list[str] = []
    target_table = tables[-1]
    for edge in graph.edges_from(tables[0]):
        if edge.target_table == target_table:
            candidate_path = f"{tables[0]}.{target_table}"
            if candidate_path != path_id:
                shortcuts.append(candidate_path)
    return sorted(set(shortcuts))


def _build_difficulty_features(
    graph: SchemaGraph,
    *,
    edges: list[ForeignKeyEdge],
    shortcut_candidates: list[str],
) -> DifficultyFeatures:
    fanouts = [edge.fanout_estimate or 1.0 for edge in edges]
    fanout_product = 1.0
    for fanout in fanouts:
        fanout_product *= fanout

    nullable_hop = False
    cardinality_estimate: float | None = None
    if edges:
        target_table = graph.get_table(edges[-1].target_table, schema_name=edges[-1].target_schema)
        if target_table.row_estimate is not None:
            cardinality_estimate = float(target_table.row_estimate)
        for edge in edges:
            source_table = graph.get_table(edge.source_table, schema_name=edge.source_schema)
            if any(
                source_table.get_column(column_name).is_nullable
                for column_name in edge.source_columns
            ):
                nullable_hop = True
                break

    return DifficultyFeatures(
        required_hops=len(edges),
        fanout_max=max(fanouts) if fanouts else 1.0,
        fanout_product=fanout_product,
        cardinality_estimate=cardinality_estimate,
        has_unique_join=any(edge.source_is_unique for edge in edges),
        has_nullable_hop=nullable_hop,
        shortcut_count=len(shortcut_candidates),
    )
