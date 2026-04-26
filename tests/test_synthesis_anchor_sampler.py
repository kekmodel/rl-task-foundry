from __future__ import annotations

import random
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.synthesis.anchor_sampler import (
    AnchorTableCandidate,
    build_anchor_table_candidates,
    score_anchor_candidate,
    select_anchor_tables,
    summarize_anchor_candidate_pool,
)
from rl_task_foundry.synthesis.synthesis_db import SynthesisDb


def _column(
    table: str,
    name: str,
    data_type: str = "text",
    *,
    ordinal: int = 1,
    visibility: str = "user_visible",
    primary_key: bool = False,
    foreign_key: bool = False,
) -> ColumnProfile:
    return ColumnProfile(
        schema_name="public",
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=ordinal,
        is_nullable=False,
        visibility=visibility,  # type: ignore[arg-type]
        is_primary_key=primary_key,
        is_foreign_key=foreign_key,
    )


def _table(
    name: str,
    columns: list[ColumnProfile],
    *,
    pk: tuple[str, ...],
    rows: int | None = 100,
) -> TableProfile:
    return TableProfile(
        schema_name="public",
        table_name=name,
        columns=columns,
        primary_key=pk,
        row_estimate=rows,
    )


def _config_with_output(tmp_path: Path):
    config = load_config("rl_task_foundry.yaml")
    return config.model_copy(
        update={
            "output": OutputConfig(
                run_db_path=tmp_path / "run.db",
                traces_dir=tmp_path / "traces",
            )
        },
        deep=True,
    )


def test_anchor_table_candidates_do_not_require_large_single_column_pk() -> None:
    graph = SchemaGraph(
        tables=[
            _table(
                "order_line",
                [
                    _column(
                        "order_line",
                        "order_id",
                        "integer",
                        primary_key=True,
                    ),
                    _column(
                        "order_line",
                        "line_no",
                        "integer",
                        ordinal=2,
                        primary_key=True,
                    ),
                    _column("order_line", "status", ordinal=3),
                ],
                pk=("order_id", "line_no"),
                rows=3,
            ),
            _table(
                "pure_bridge",
                [
                    _column(
                        "pure_bridge",
                        "left_id",
                        "integer",
                        primary_key=True,
                        foreign_key=True,
                    ),
                    _column(
                        "pure_bridge",
                        "right_id",
                        "integer",
                        ordinal=2,
                        primary_key=True,
                        foreign_key=True,
                    ),
                ],
                pk=("left_id", "right_id"),
                rows=500,
            ),
        ],
        edges=[],
    )

    candidates = build_anchor_table_candidates(graph)
    by_table = {candidate.table.table_name: candidate for candidate in candidates}

    assert "order_line" in by_table
    assert "pure_bridge" in by_table
    assert by_table["order_line"].score > by_table["pure_bridge"].score


def test_anchor_candidate_metrics_score_rlvr_start_value() -> None:
    strong = {
        "qualified_table": "public.customer",
        "row_id": 284,
        "preview": {"display_name": "Mary Smith", "status": "active"},
        "relationship_summary": [
            {"path": "public.customer <- public.rental.customer_id", "count": 18}
        ],
    }
    weak = {
        "qualified_table": "public.phone",
        "row_id": 1,
        "preview": {},
        "relationship_summary": [],
    }

    strong_metrics = score_anchor_candidate(strong)
    weak_metrics = score_anchor_candidate(weak)
    summary = summarize_anchor_candidate_pool(
        [strong, weak],
        eligible_table_count=4,
    )

    assert strong_metrics["preferred"] is True
    assert strong_metrics["rlvr_start_score"] > 0.85
    assert weak_metrics["preferred"] is False
    assert weak_metrics["dead_anchor"] is True
    assert weak_metrics["id_one_like"] is True
    assert summary["candidate_count"] == 2
    assert summary["dead_anchor_rate"] == 0.5
    assert summary["id_one_like_rate"] == 0.5
    assert summary["rlvr_start_pool_score"] < 0.8


def test_anchor_table_selection_does_not_force_low_score_structural_roles() -> None:
    strong_table = _table(
        "customer",
        [
            _column("customer", "customer_id", "integer", primary_key=True),
            _column("customer", "display_name"),
        ],
        pk=("customer_id",),
    )
    weak_table = _table(
        "phone",
        [_column("phone", "phone_id", "integer", primary_key=True)],
        pk=("phone_id",),
    )
    selected = select_anchor_tables(
        [
            AnchorTableCandidate(
                table=weak_table,
                score=0.1,
                structure="structural",
                preview_columns=(),
                incoming_edges=(),
                outgoing_edges=(),
            ),
            AnchorTableCandidate(
                table=strong_table,
                score=100.0,
                structure="hub",
                preview_columns=("display_name",),
                incoming_edges=(),
                outgoing_edges=(),
            ),
        ],
        limit=1,
        rng=random.Random(0),
    )

    assert selected[0].table.table_name == "customer"


@dataclass(slots=True)
class _FakeConnection:
    fetchrow_sql: list[str] = field(default_factory=list)
    fetchval_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    async def fetchrow(self, sql: str, *params: object):
        del params
        self.fetchrow_sql.append(sql)
        if '"customer"' in sql:
            return {
                "customer_id": 284,
                "display_name": "Mary Smith",
                "created_at": "2025-01-03",
            }
        return None

    async def fetchval(self, sql: str, *params: object):
        self.fetchval_calls.append((sql, params))
        if '"rental"' in sql:
            return 18
        return 0


@dataclass(slots=True)
class _SequentialConnection:
    rows: list[dict[str, object]]
    fetchrow_sql: list[str] = field(default_factory=list)

    async def fetchrow(self, sql: str, *params: object):
        del params
        self.fetchrow_sql.append(sql)
        if self.rows:
            return self.rows.pop(0)
        return None

    async def fetchval(self, sql: str, *params: object):
        del sql, params
        return 18


@dataclass(slots=True)
class _FakePools:
    conn: object

    @asynccontextmanager
    async def control_connection(self):
        yield self.conn


@pytest.mark.asyncio
async def test_random_anchor_candidates_include_preview_and_relation_counts(
    tmp_path: Path,
) -> None:
    customer = _table(
        "customer",
        [
            _column(
                "customer",
                "customer_id",
                "integer",
                visibility="blocked",
                primary_key=True,
            ),
            _column("customer", "display_name", ordinal=2),
            _column("customer", "created_at", "timestamp", ordinal=3),
        ],
        pk=("customer_id",),
        rows=100,
    )
    rental = _table(
        "rental",
        [
            _column(
                "rental",
                "rental_id",
                "integer",
                visibility="blocked",
                primary_key=True,
            ),
            _column(
                "rental",
                "customer_id",
                "integer",
                ordinal=2,
                visibility="blocked",
                foreign_key=True,
            ),
        ],
        pk=("rental_id",),
        rows=0,
    )
    graph = SchemaGraph(
        tables=[customer, rental],
        edges=[
            ForeignKeyEdge(
                constraint_name="rental_customer_fk",
                source_schema="public",
                source_table="rental",
                source_columns=("customer_id",),
                target_schema="public",
                target_table="customer",
                target_columns=("customer_id",),
            )
        ],
    )
    conn = _FakeConnection()
    synthesis_db = SynthesisDb(
        db_id="test",
        config=_config_with_output(tmp_path),
        database_pools=_FakePools(conn),  # type: ignore[arg-type]
    )
    synthesis_db.adopt_schema_graph(graph)

    hint = await synthesis_db.random_anchor_candidates(limit=1)

    assert hint is not None
    candidates = hint["candidate_entities"]
    assert isinstance(candidates, list)
    quality_metrics = candidates[0].pop("quality_metrics")
    assert candidates == [
        {
            "table": "customer",
            "qualified_table": "public.customer",
            "pk_columns": ["customer_id"],
            "row_id": 284,
            "entity": {"customer_id": 284},
            "preview": {
                "display_name": "Mary Smith",
                "created_at": "2025-01-03",
            },
            "relationship_summary": [
                {
                    "path": "public.customer <- public.rental.customer_id",
                    "direction": "incoming",
                    "count": 18,
                }
            ],
        }
    ]
    assert isinstance(quality_metrics, dict)
    assert quality_metrics["preferred"] is True
    assert quality_metrics["rlvr_start_score"] > 0.85
    assert candidates[0]["row_id"] != 1
    assert any("TABLESAMPLE SYSTEM" in sql for sql in conn.fetchrow_sql)
    assert conn.fetchval_calls


@pytest.mark.asyncio
async def test_random_anchor_candidates_resample_first_id_when_better_row_exists(
    tmp_path: Path,
) -> None:
    customer = _table(
        "customer",
        [
            _column(
                "customer",
                "customer_id",
                "integer",
                visibility="blocked",
                primary_key=True,
            ),
            _column("customer", "display_name", ordinal=2),
        ],
        pk=("customer_id",),
        rows=100,
    )
    graph = SchemaGraph(tables=[customer], edges=[])
    conn = _SequentialConnection(
        rows=[
            {"customer_id": 1, "display_name": "First Row"},
            {"customer_id": 284, "display_name": "Better Row"},
        ]
    )
    synthesis_db = SynthesisDb(
        db_id="test",
        config=_config_with_output(tmp_path),
        database_pools=_FakePools(conn),  # type: ignore[arg-type]
    )
    synthesis_db.adopt_schema_graph(graph)

    hint = await synthesis_db.random_anchor_candidates(limit=1)

    assert hint is not None
    candidates = hint["candidate_entities"]
    assert isinstance(candidates, list)
    assert candidates[0]["row_id"] == 284
    assert len(conn.fetchrow_sql) == 2
