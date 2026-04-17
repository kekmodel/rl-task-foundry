"""Per-database cache and resource holder for synthesis.

A ``SynthesisDb`` owns the long-lived, per-``db_id`` artifacts that
synthesis needs but that should not be rebuilt for every trial: schema
introspection, data profile, schema snapshot (materialized to disk for
bundle export), and the random-anchor sampler. Top-level orchestrators
(HarvestRunner, RealDbTrialRunner, ProofTaskRunner, SynthesisRegistryRunner)
own one ``SynthesisDb`` per database and inject it into
``SynthesisAgentRuntime``; many synthesis conversations for the same
``db_id`` share a single ``SynthesisDb``.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.db import DatabasePools, ensure_attached_database_pools
from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.profiler import DataProfile, profile_database
from rl_task_foundry.synthesis.snapshot_materializer import (
    SchemaSnapshotMaterializer,
)
from rl_task_foundry.tooling.common import SchemaSnapshot, snapshot_from_graph

logger = logging.getLogger(__name__)


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

    async def random_anchor(self) -> dict[str, object] | None:
        graph = await self.schema_graph()
        hub_tables = [
            t
            for t in graph.tables
            if t.primary_key and len(t.primary_key) == 1 and (t.row_estimate or 0) >= 100
        ]
        if not hub_tables:
            return None
        table = random.choice(hub_tables)
        pk_col = table.primary_key[0]
        try:
            pools = await self.ensure_database_pools()
            async with pools.control_connection() as conn:
                row = await conn.fetchrow(
                    f"SELECT {pk_col} FROM {table.qualified_name} "
                    f"ORDER BY random() LIMIT 1"
                )
                if row:
                    return {pk_col: row[pk_col]}
        except Exception:
            logger.warning(
                "anchor seeding failed for %s",
                table.qualified_name,
                exc_info=True,
            )
        return None

    async def close(self) -> None:
        if self._database_pools is not None:
            if self._owns_database_pools:
                await self._database_pools.close()
            self._database_pools = None
