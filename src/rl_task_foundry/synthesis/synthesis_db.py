"""Per-database cache and resource holder for synthesis.

A ``SynthesisDb`` owns the long-lived, per-``db_id`` artifacts that synthesis
needs but that should not be rebuilt for every trial: schema introspection,
data profile, atomic-tool bundle, materialized tool executors, and the
random-anchor sampler. Top-level orchestrators (HarvestRunner, RealDbTrialRunner,
ProofTaskRunner, SynthesisRegistryRunner) own one ``SynthesisDb`` per database
and inject it into ``SynthesisAgentRuntime``; many synthesis conversations for
the same ``db_id`` share a single ``SynthesisDb``.
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
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle, AtomicToolGenerator
from rl_task_foundry.synthesis.tool_runtime import (
    ToolExecutor,
    bind_atomic_tool_executor,
    load_atomic_tool_module,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SynthesisDb:
    db_id: str
    config: AppConfig
    database_pools: DatabasePools | None = None
    atomic_tool_materializer: AtomicToolMaterializer | None = None
    _database_pools: DatabasePools | None = field(default=None, init=False, repr=False)
    _owns_database_pools: bool = field(default=True, init=False, repr=False)
    _atomic_tool_materializer: AtomicToolMaterializer | None = field(
        default=None, init=False, repr=False
    )
    _graph_cache: SchemaGraph | None = field(default=None, init=False, repr=False)
    _data_profile_cache: DataProfile | None = field(default=None, init=False, repr=False)
    _atomic_tool_bundle: AtomicToolBundle | None = field(default=None, init=False, repr=False)
    _tool_executors: dict[str, ToolExecutor] | None = field(default=None, init=False, repr=False)
    _graph_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _atomic_tool_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.database_pools is not None:
            self._database_pools = self.database_pools
            self._owns_database_pools = False
        if self.atomic_tool_materializer is not None:
            self._atomic_tool_materializer = self.atomic_tool_materializer
        else:
            self._atomic_tool_materializer = AtomicToolMaterializer.for_config(self.config)

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

    async def data_profile(self) -> DataProfile:
        if self._data_profile_cache is not None:
            return self._data_profile_cache
        graph = await self.schema_graph()
        self._data_profile_cache = await profile_database(self.config.database, graph)
        return self._data_profile_cache

    async def atomic_tool_bundle(self) -> AtomicToolBundle:
        if self._atomic_tool_bundle is not None:
            return self._atomic_tool_bundle
        async with self._atomic_tool_lock:
            if self._atomic_tool_bundle is not None:
                return self._atomic_tool_bundle
            graph = await self.schema_graph()
            bundle = AtomicToolGenerator(self.config.atomic_tools).generate_bundle(
                graph, db_id=self.db_id
            )
            assert self._atomic_tool_materializer is not None
            self._atomic_tool_materializer.materialize_bundle(bundle)
            self._atomic_tool_bundle = bundle
        return bundle

    async def tool_executors(self) -> dict[str, ToolExecutor]:
        if self._tool_executors is not None:
            return self._tool_executors
        bundle = await self.atomic_tool_bundle()
        pools = await self.database_pools_for_tools()
        assert self._atomic_tool_materializer is not None
        materialization = self._atomic_tool_materializer.materialize_bundle(bundle)
        module = load_atomic_tool_module(
            materialization.source_path,
            module_name=f"rl_task_foundry_synthesis_atomic_tools_{self.db_id}",
        )
        resolved = {
            tool.name: bind_atomic_tool_executor(
                module=module,
                tool_name=tool.name,
                pools=pools,
            )
            for tool in bundle.tools
        }
        self._tool_executors = resolved
        return resolved

    async def database_pools_for_tools(self) -> DatabasePools:
        return await ensure_attached_database_pools(
            self,
            attr_name="_database_pools",
            config=self.config.database,
        )

    def adopt_schema_graph(self, graph: SchemaGraph) -> None:
        """Inject a pre-built schema graph (e.g. from the registry entry)."""

        self._graph_cache = graph

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
            pools = await self.database_pools_for_tools()
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
        self._tool_executors = None
        self._atomic_tool_bundle = None
