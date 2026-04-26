"""Per-database cache and resource holder for synthesis.

A ``SynthesisDb`` owns the long-lived, per-``db_id`` artifacts that
synthesis needs but that should not be rebuilt for every trial: schema
introspection, data profile, schema snapshot (materialized to disk for
bundle export), and the random-anchor sampler. Top-level orchestrators
(HarvestRunner, RealDbTrialRunner, SynthesisRegistryRunner, and the proof task)
own one ``SynthesisDb`` per database and inject it into
``SynthesisAgentRuntime``; many synthesis conversations for the same
``db_id`` share a single ``SynthesisDb``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import random
from dataclasses import dataclass, field
from decimal import Decimal

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.db import DatabasePools, ensure_attached_database_pools
from rl_task_foundry.schema.graph import ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.profiler import DataProfile, profile_database
from rl_task_foundry.synthesis.anchor_sampler import (
    AnchorTableCandidate,
    build_anchor_table_candidates,
    score_anchor_candidate,
    select_anchor_tables,
)
from rl_task_foundry.synthesis.snapshot_materializer import (
    SchemaSnapshotMaterializer,
)
from rl_task_foundry.tooling.common import SchemaSnapshot, snapshot_from_graph
from rl_task_foundry.tooling.common.sql import quote_ident, quote_table, readonly_select

logger = logging.getLogger(__name__)
_ANCHOR_ROW_SAMPLE_ATTEMPTS = 2


@dataclass(slots=True)
class SynthesisDb:
    db_id: str
    config: AppConfig
    database_pools: DatabasePools | None = None
    snapshot_materializer: SchemaSnapshotMaterializer | None = None
    _database_pools: DatabasePools | None = field(default=None, init=False, repr=False)
    _owns_database_pools: bool = field(default=True, init=False, repr=False)
    _snapshot_materializer: SchemaSnapshotMaterializer | None = field(
        default=None, init=False, repr=False
    )
    _graph_cache: SchemaGraph | None = field(default=None, init=False, repr=False)
    _schema_snapshot_cache: SchemaSnapshot | None = field(
        default=None, init=False, repr=False
    )
    _data_profile_cache: DataProfile | None = field(default=None, init=False, repr=False)
    _graph_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.database_pools is not None:
            self._database_pools = self.database_pools
            self._owns_database_pools = False
        if self.snapshot_materializer is not None:
            self._snapshot_materializer = self.snapshot_materializer
        else:
            self._snapshot_materializer = SchemaSnapshotMaterializer.for_config(
                self.config
            )

    async def schema_graph(self) -> SchemaGraph:
        if self._graph_cache is not None:
            return self._graph_cache
        async with self._graph_lock:
            if self._graph_cache is not None:
                return self._graph_cache
            introspector = PostgresSchemaIntrospector(
                database=self.config.database,
                default_visibility=self.config.privacy.default_visibility,
                visibility_overrides=self.config.privacy.visibility_overrides,
            )
            self._graph_cache = await introspector.introspect()
        return self._graph_cache

    async def schema_snapshot(self) -> SchemaSnapshot:
        """Immutable snapshot of the schema graph for tooling callers.

        Derived from the cached `SchemaGraph` via `snapshot_from_graph`
        and cached per `SynthesisDb`. On first resolution the snapshot
        is also materialized to disk (see `SchemaSnapshotMaterializer`)
        so `bundle_exporter` can ship it without touching the DB again.
        """
        if self._schema_snapshot_cache is not None:
            return self._schema_snapshot_cache
        graph = await self.schema_graph()
        snapshot = snapshot_from_graph(graph)
        assert self._snapshot_materializer is not None
        self._snapshot_materializer.materialize(db_id=self.db_id, snapshot=snapshot)
        self._schema_snapshot_cache = snapshot
        return self._schema_snapshot_cache

    async def data_profile(self) -> DataProfile:
        if self._data_profile_cache is not None:
            return self._data_profile_cache
        graph = await self.schema_graph()
        self._data_profile_cache = await profile_database(self.config.database, graph)
        return self._data_profile_cache

    async def ensure_database_pools(self) -> DatabasePools:
        """Lazily attach a DatabasePools handle and return it.

        Used by the runtime for composer-tool connections and by
        `random_anchor` for control-plane queries.
        """
        return await ensure_attached_database_pools(
            self,
            attr_name="_database_pools",
            config=self.config.database,
        )

    def adopt_schema_graph(self, graph: SchemaGraph) -> None:
        """Inject a pre-built schema graph (e.g. from the registry entry)."""

        self._graph_cache = graph
        self._schema_snapshot_cache = None

    def use_snapshot_materializer(
        self,
        materializer: SchemaSnapshotMaterializer,
    ) -> None:
        """Redirect materialized schema artifacts to a new output root."""

        self._snapshot_materializer = materializer
        if self._schema_snapshot_cache is not None:
            materializer.materialize(
                db_id=self.db_id,
                snapshot=self._schema_snapshot_cache,
            )

    async def random_anchor_candidates(
        self,
        *,
        limit: int = 10,
        rng: random.Random | None = None,
    ) -> dict[str, object] | None:
        """Return diverse observed anchor candidates for one synthesis episode.

        Candidates are not answer hints. They randomize the episode's initial
        entity surface so a stateless composer does not repeatedly start from
        the first or smallest id in an arbitrary database.
        """

        if limit <= 0:
            return None
        graph = await self.schema_graph()
        table_candidates = build_anchor_table_candidates(graph)
        if not table_candidates:
            return None
        selected = select_anchor_tables(
            table_candidates,
            limit=max(limit * 4, limit),
            rng=rng,
        )
        snapshot = await self.schema_snapshot()
        candidates: list[dict[str, object]] = []
        try:
            pools = await self.ensure_database_pools()
            async with pools.control_connection() as conn:
                for table_candidate in selected:
                    candidate = await self._build_anchor_candidate(
                        conn,
                        snapshot=snapshot,
                        table_candidate=table_candidate,
                    )
                    if candidate is None:
                        continue
                    quality_metrics = score_anchor_candidate(candidate)
                    candidate["quality_metrics"] = quality_metrics
                    candidates.append(candidate)
                    if len(candidates) >= limit:
                        break
        except Exception:
            logger.warning(
                "anchor candidate seeding failed for %s",
                self.db_id,
                exc_info=True,
            )
        if not candidates:
            return None
        return {
            "candidate_entities": candidates,
            "selection_note": (
                "Random observed starting entities. Use one if helpful; "
                "ignore them and inspect the DB if none fit the draft."
            ),
        }

    async def random_anchor(self) -> dict[str, object] | None:
        anchor_hint = await self.random_anchor_candidates(limit=1)
        if not anchor_hint:
            return None
        candidates = anchor_hint.get("candidate_entities")
        if not isinstance(candidates, list) or not candidates:
            return None
        first = candidates[0]
        if not isinstance(first, dict):
            return None
        pk_columns = first.get("pk_columns")
        entity = first.get("entity")
        if not isinstance(pk_columns, list) or not pk_columns:
            return first
        if not isinstance(entity, dict):
            return first
        pk_column = pk_columns[0]
        row_id = first.get("row_id")
        return {
            "table": first.get("table"),
            "qualified_table": first.get("qualified_table"),
            "pk_column": pk_column,
            "pk_columns": pk_columns,
            "row_id": row_id,
            "entity": entity,
            "preview": first.get("preview", {}),
            "relationship_summary": first.get("relationship_summary", []),
        }

    async def _build_anchor_candidate(
        self,
        conn: object,
        *,
        snapshot: SchemaSnapshot,
        table_candidate: AnchorTableCandidate,
    ) -> dict[str, object] | None:
        table = table_candidate.table
        selected_columns = _anchor_select_columns(table, table_candidate)
        best_candidate: dict[str, object] | None = None
        best_score = -1.0
        for _ in range(_ANCHOR_ROW_SAMPLE_ATTEMPTS):
            row = await _fetch_random_row(conn, table=table, columns=selected_columns)
            if not row:
                continue
            candidate = await _anchor_candidate_from_row(
                conn,
                snapshot=snapshot,
                table_candidate=table_candidate,
                row=row,
            )
            if candidate is None:
                continue
            metrics = score_anchor_candidate(candidate)
            score = float(metrics["rlvr_start_score"])
            if score > best_score:
                best_score = score
                best_candidate = candidate
            if metrics.get("preferred") is True and metrics.get("id_one_like") is False:
                break
        return best_candidate

    async def close(self) -> None:
        if self._database_pools is not None:
            if self._owns_database_pools:
                await self._database_pools.close()
            self._database_pools = None


async def _anchor_candidate_from_row(
    conn: object,
    *,
    snapshot: SchemaSnapshot,
    table_candidate: AnchorTableCandidate,
    row: object,
) -> dict[str, object] | None:
    table = table_candidate.table
    entity = _anchor_entity(row, table)
    if entity is None:
        return None
    row_id = _row_id_from_entity(entity, table)
    preview: dict[str, object] = {}
    for column in table_candidate.preview_columns:
        value = _row_value(row, column)
        if value is not None:
            preview[column] = _json_safe(value)
    relationship_summary = await _relationship_summary(
        conn,
        table=table,
        row=row,
        edges=[
            *table_candidate.incoming_edges,
            *table_candidate.outgoing_edges,
        ],
    )
    table_handle = snapshot.table(table.qualified_name).handle
    return {
        "table": table_handle,
        "qualified_table": table.qualified_name,
        "pk_columns": list(table.primary_key),
        "row_id": row_id,
        "entity": entity,
        "preview": preview,
        "relationship_summary": relationship_summary,
    }


def _anchor_select_columns(
    table: TableProfile,
    table_candidate: AnchorTableCandidate,
) -> list[str]:
    columns: list[str] = []
    relationship_columns: list[str] = []
    for edge in table_candidate.incoming_edges:
        relationship_columns.extend(edge.target_columns)
    for edge in table_candidate.outgoing_edges:
        relationship_columns.extend(edge.source_columns)
    for column in (
        *table.primary_key,
        *table_candidate.preview_columns,
        *relationship_columns,
    ):
        if column not in columns:
            columns.append(column)
    return columns


async def _fetch_random_row(
    conn: object,
    *,
    table: TableProfile,
    columns: list[str],
) -> object | None:
    select_expr = ", ".join(quote_ident(column) for column in columns)
    quoted_table = quote_table(table.schema_name, table.table_name)
    try:
        row = await conn.fetchrow(
            readonly_select(
                f"SELECT {select_expr} FROM {quoted_table} "
                "TABLESAMPLE SYSTEM (1) LIMIT 1"
            )
        )
        if row:
            return row
    except Exception:
        logger.debug(
            "TABLESAMPLE anchor row fetch failed for %s; falling back",
            table.qualified_name,
            exc_info=True,
        )
    return await conn.fetchrow(
        readonly_select(
            f"SELECT {select_expr} FROM {quoted_table} ORDER BY random() LIMIT 1"
        )
    )


def _anchor_entity(row: object, table: TableProfile) -> dict[str, object] | None:
    entity: dict[str, object] = {}
    for column in table.primary_key:
        value = _row_value(row, column)
        if value is None:
            return None
        entity[column] = _json_safe(value)
    return entity


def _row_id_from_entity(entity: dict[str, object], table: TableProfile) -> object:
    values = [entity[column] for column in table.primary_key]
    if len(values) == 1:
        return values[0]
    return values


async def _relationship_summary(
    conn: object,
    *,
    table: TableProfile,
    row: object,
    edges: list[ForeignKeyEdge],
    limit: int = 6,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for edge in edges[:limit]:
        summary = await _edge_count(conn, table=table, row=row, edge=edge)
        if summary is not None:
            summaries.append(summary)
    return summaries


async def _edge_count(
    conn: object,
    *,
    table: TableProfile,
    row: object,
    edge: ForeignKeyEdge,
) -> dict[str, object] | None:
    if edge.target_schema == table.schema_name and edge.target_table == table.table_name:
        values = [_row_value(row, column) for column in edge.target_columns]
        if any(value is None for value in values):
            return None
        count = await _count_matching_rows(
            conn,
            schema=edge.source_schema,
            table=edge.source_table,
            columns=edge.source_columns,
            values=values,
        )
        return {
            "path": _incoming_edge_label(edge),
            "direction": "incoming",
            "count": count,
        }
    if edge.source_schema == table.schema_name and edge.source_table == table.table_name:
        values = [_row_value(row, column) for column in edge.source_columns]
        if any(value is None for value in values):
            return None
        count = await _count_matching_rows(
            conn,
            schema=edge.target_schema,
            table=edge.target_table,
            columns=edge.target_columns,
            values=values,
        )
        return {
            "path": _outgoing_edge_label(edge),
            "direction": "outgoing",
            "count": count,
        }
    return None


async def _count_matching_rows(
    conn: object,
    *,
    schema: str,
    table: str,
    columns: tuple[str, ...],
    values: list[object],
) -> int:
    where = " AND ".join(
        f"{quote_ident(column)} = ${index}"
        for index, column in enumerate(columns, start=1)
    )
    value = await conn.fetchval(
        readonly_select(
            f"SELECT count(*) FROM {quote_table(schema, table)} WHERE {where}"
        ),
        *values,
    )
    return int(value or 0)


def _row_value(row: object, column: str) -> object:
    try:
        return row[column]  # type: ignore[index]
    except (KeyError, TypeError):
        return getattr(row, column, None)


def _json_safe(value: object) -> object:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, _dt.datetime | _dt.date | _dt.time):
        return value.isoformat()
    return str(value)


def _incoming_edge_label(edge: ForeignKeyEdge) -> str:
    return (
        f"{edge.target_qualified_name} <- {edge.source_qualified_name}."
        f"{','.join(edge.source_columns)}"
    )


def _outgoing_edge_label(edge: ForeignKeyEdge) -> str:
    return (
        f"{edge.source_qualified_name}.{','.join(edge.source_columns)} -> "
        f"{edge.target_qualified_name}"
    )
