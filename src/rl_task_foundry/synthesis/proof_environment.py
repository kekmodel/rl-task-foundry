"""Synthetic proof fixture smoke slice.

The proof fixture drives the full synthesis-through-registry pipeline
(composer conversation, solver rollout, quality gate, registry commit) without
consuming provider quota.

Shape:
1. Provision an ephemeral Postgres schema from the packaged fixture DDL/seed.
2. Build an ``AppConfig`` override that points the schema allowlist and
   calibration band at the ephemeral schema and keeps rollouts small.
3. Inject a ``ScriptedComposerBackend`` that replays a fixed sequence of
   composer-tool observations and submits the canonical itinerary draft.
4. Inject a solver runtime that alternates matched/unmatched answers so the
   observed pass rate lands inside the quality-gate band.
5. Delegate to ``RealDbTrialRunner`` so smoke behavior matches real-DB trials.
6. Drop the ephemeral schema on teardown.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg

from rl_task_foundry.config.models import AppConfig, DatabaseConfig
from rl_task_foundry.infra.db import (
    DatabasePools,
    _apply_session_settings,
    mutating_control_session_settings,
)
from rl_task_foundry.pipeline.solver_orchestrator import SolverOrchestrator
from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.solver.runtime import AgentRuntime, SolverEpisodeInput
from rl_task_foundry.synthesis.backend_scripted import (
    ScriptedAtomicToolCall,
    ScriptedComposerBackend,
    ScriptedComposerScript,
)
from rl_task_foundry.synthesis.contracts import TaskBundleContract
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
    RealDbTrialSummary,
)
from rl_task_foundry.synthesis.submit_draft_tool import SubmitDraftPayload
from rl_task_foundry.synthesis.synthesis_db import SynthesisDb
from rl_task_foundry.tooling.common.sql import quote_ident

PROOF_DB_ID = "proof_trip_fixture"
PROOF_TASK_TOPIC = "itinerary"


PROOF_ANCHOR_ENTITY: dict[str, object] = {"anchor_id": 1}


PROOF_CANONICAL_ANSWER: list[dict[str, object]] = [
    {
        "day": 1,
        "city": "Seoul",
        "lodging": "Seoul Station Stay",
        "activity": "Han River Night Walk",
        "total_cost": 180,
    },
    {
        "day": 2,
        "city": "Suwon",
        "lodging": "Suwon Fortress Hotel",
        "activity": "Fortress Loop Tour",
        "total_cost": 160,
    },
    {
        "day": 3,
        "city": "Incheon",
        "lodging": "Incheon Harbor Inn",
        "activity": "Harbor Sunset Ferry",
        "total_cost": 170,
    },
]


PROOF_QUESTION_BODY = (
    "봄 시즌 3일 출장 일정표를 만들어 주세요. 각 day의 city, lodging, "
    "activity, total_cost를 포함해야 하고 각 day는 하나의 city를 방문해야 하며 "
    "전체 itinerary에서 city는 중복되면 안 됩니다. day별 total_cost는 250 이하여야 "
    "하며, 연속된 day의 city는 인접한 지역이어야 합니다. 가능한 일정이 여러 개면 "
    "day 1의 city 이름, 그다음 day 2의 city 이름, 그다음 day 3의 city 이름의 "
    "사전순이 가장 앞서는 답을 고르세요."
)


PROOF_FIXTURE_DDL = """
CREATE TABLE proof_anchors (
    anchor_id INTEGER PRIMARY KEY,
    season TEXT NOT NULL,
    budget_bucket TEXT NOT NULL,
    start_city_id INTEGER NOT NULL
);

CREATE TABLE proof_cities (
    city_id INTEGER PRIMARY KEY,
    city_name TEXT NOT NULL,
    region_name TEXT NOT NULL,
    season TEXT NOT NULL
);

CREATE TABLE proof_city_links (
    city_id INTEGER NOT NULL,
    neighbor_city_id INTEGER NOT NULL,
    PRIMARY KEY (city_id, neighbor_city_id),
    FOREIGN KEY (city_id) REFERENCES proof_cities(city_id),
    FOREIGN KEY (neighbor_city_id) REFERENCES proof_cities(city_id)
);

CREATE TABLE proof_lodgings (
    lodging_id INTEGER PRIMARY KEY,
    city_id INTEGER NOT NULL,
    lodging_name TEXT NOT NULL,
    nightly_cost INTEGER NOT NULL,
    FOREIGN KEY (city_id) REFERENCES proof_cities(city_id)
);

CREATE TABLE proof_activities (
    activity_id INTEGER PRIMARY KEY,
    city_id INTEGER NOT NULL,
    activity_name TEXT NOT NULL,
    ticket_cost INTEGER NOT NULL,
    FOREIGN KEY (city_id) REFERENCES proof_cities(city_id)
);
""".strip()


PROOF_FIXTURE_SEED = """
INSERT INTO proof_anchors (anchor_id, season, budget_bucket, start_city_id) VALUES
    (1, 'spring', 'mid', 101);

INSERT INTO proof_cities (city_id, city_name, region_name, season) VALUES
    (101, 'Seoul', 'capital', 'spring'),
    (102, 'Suwon', 'capital_belt', 'spring'),
    (103, 'Incheon', 'capital_belt', 'spring'),
    (104, 'Gangneung', 'east_coast', 'summer');

INSERT INTO proof_city_links (city_id, neighbor_city_id) VALUES
    (101, 102),
    (102, 101),
    (102, 103),
    (103, 102);

INSERT INTO proof_lodgings (lodging_id, city_id, lodging_name, nightly_cost) VALUES
    (201, 101, 'Seoul Station Stay', 110),
    (202, 102, 'Suwon Fortress Hotel', 100),
    (203, 103, 'Incheon Harbor Inn', 95);

INSERT INTO proof_activities (activity_id, city_id, activity_name, ticket_cost) VALUES
    (301, 101, 'Han River Night Walk', 70),
    (302, 102, 'Fortress Loop Tour', 60),
    (303, 103, 'Harbor Sunset Ferry', 75);
""".strip()


def _split_sql_statements(source: str) -> Iterator[str]:
    for raw in source.split(";"):
        cleaned = raw.strip()
        if cleaned:
            yield cleaned


def _canonical_answer_json() -> str:
    return json.dumps(PROOF_CANONICAL_ANSWER, ensure_ascii=False)


def _unmatched_answer_json() -> str:
    return json.dumps(
        [
            {
                "day": 1,
                "city": "Gangneung",
                "lodging": "Seoul Station Stay",
                "activity": "Han River Night Walk",
                "total_cost": 999,
            }
        ],
        ensure_ascii=False,
    )


def build_proof_question() -> str:
    return PROOF_QUESTION_BODY


def build_proof_schema_graph(schema_name: str) -> SchemaGraph:
    return SchemaGraph(
        tables=[
            TableProfile(
                schema_name=schema_name,
                table_name="proof_anchors",
                columns=[
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_anchors",
                        column_name="anchor_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_anchors",
                        column_name="season",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_anchors",
                        column_name="budget_bucket",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_anchors",
                        column_name="start_city_id",
                        data_type="integer",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="user_visible",
                        is_foreign_key=True,
                    ),
                ],
                primary_key=("anchor_id",),
            ),
            TableProfile(
                schema_name=schema_name,
                table_name="proof_cities",
                columns=[
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_cities",
                        column_name="city_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_cities",
                        column_name="city_name",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_cities",
                        column_name="region_name",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_cities",
                        column_name="season",
                        data_type="text",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
                primary_key=("city_id",),
            ),
            TableProfile(
                schema_name=schema_name,
                table_name="proof_city_links",
                columns=[
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_city_links",
                        column_name="city_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_city_links",
                        column_name="neighbor_city_id",
                        data_type="integer",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                        is_foreign_key=True,
                    ),
                ],
                primary_key=("city_id", "neighbor_city_id"),
            ),
            TableProfile(
                schema_name=schema_name,
                table_name="proof_lodgings",
                columns=[
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_lodgings",
                        column_name="lodging_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_lodgings",
                        column_name="city_id",
                        data_type="integer",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_lodgings",
                        column_name="lodging_name",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_lodgings",
                        column_name="nightly_cost",
                        data_type="integer",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
                primary_key=("lodging_id",),
            ),
            TableProfile(
                schema_name=schema_name,
                table_name="proof_activities",
                columns=[
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_activities",
                        column_name="activity_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_activities",
                        column_name="city_id",
                        data_type="integer",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_activities",
                        column_name="activity_name",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name=schema_name,
                        table_name="proof_activities",
                        column_name="ticket_cost",
                        data_type="integer",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
                primary_key=("activity_id",),
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="anchors_start_city",
                source_schema=schema_name,
                source_table="proof_anchors",
                source_columns=("start_city_id",),
                target_schema=schema_name,
                target_table="proof_cities",
                target_columns=("city_id",),
            ),
            ForeignKeyEdge(
                constraint_name="city_links_city",
                source_schema=schema_name,
                source_table="proof_city_links",
                source_columns=("city_id",),
                target_schema=schema_name,
                target_table="proof_cities",
                target_columns=("city_id",),
            ),
            ForeignKeyEdge(
                constraint_name="city_links_neighbor",
                source_schema=schema_name,
                source_table="proof_city_links",
                source_columns=("neighbor_city_id",),
                target_schema=schema_name,
                target_table="proof_cities",
                target_columns=("city_id",),
            ),
            ForeignKeyEdge(
                constraint_name="lodgings_city",
                source_schema=schema_name,
                source_table="proof_lodgings",
                source_columns=("city_id",),
                target_schema=schema_name,
                target_table="proof_cities",
                target_columns=("city_id",),
            ),
            ForeignKeyEdge(
                constraint_name="activities_city",
                source_schema=schema_name,
                source_table="proof_activities",
                source_columns=("city_id",),
                target_schema=schema_name,
                target_table="proof_cities",
                target_columns=("city_id",),
            ),
        ],
    )


def build_proof_composer_script() -> ScriptedComposerScript:
    atomic_tool_calls: list[ScriptedAtomicToolCall] = [
        ScriptedAtomicToolCall(
            tool_name="profile",
            params={"table": "proof_anchors"},
            result={
                "table": "proof_anchors",
                "row_count": 1,
                "columns": [
                    {
                        "name": "anchor_id",
                        "data_type": "integer",
                        "distinct_count": 1,
                        "null_count": 0,
                    },
                    {
                        "name": "season",
                        "data_type": "text",
                        "distinct_count": 1,
                        "null_count": 0,
                    },
                    {
                        "name": "budget_bucket",
                        "data_type": "text",
                        "distinct_count": 1,
                        "null_count": 0,
                    },
                ],
            },
        ),
        ScriptedAtomicToolCall(
            tool_name="sample",
            params={"table": "proof_cities", "n": 10},
            result={
                "table": "proof_cities",
                "row_count": 3,
                "rows": [
                    {"city_id": 101, "city_name": "Seoul", "region_name": "capital"},
                    {"city_id": 102, "city_name": "Suwon", "region_name": "capital_belt"},
                    {"city_id": 103, "city_name": "Incheon", "region_name": "capital_belt"},
                ],
            },
        ),
        ScriptedAtomicToolCall(
            tool_name="sample",
            params={"table": "proof_lodgings", "n": 10},
            result={
                "table": "proof_lodgings",
                "row_count": 3,
                "rows": [
                    {
                        "lodging_id": 201,
                        "city_id": 101,
                        "lodging_name": "Seoul Station Stay",
                        "nightly_cost": 110,
                    },
                    {
                        "lodging_id": 202,
                        "city_id": 102,
                        "lodging_name": "Suwon Fortress Hotel",
                        "nightly_cost": 100,
                    },
                    {
                        "lodging_id": 203,
                        "city_id": 103,
                        "lodging_name": "Incheon Harbor Inn",
                        "nightly_cost": 95,
                    },
                ],
            },
        ),
        ScriptedAtomicToolCall(
            tool_name="sample",
            params={"table": "proof_activities", "n": 10},
            result={
                "table": "proof_activities",
                "row_count": 3,
                "rows": [
                    {
                        "activity_id": 301,
                        "city_id": 101,
                        "activity_name": "Han River Night Walk",
                        "ticket_cost": 70,
                    },
                    {
                        "activity_id": 302,
                        "city_id": 102,
                        "activity_name": "Fortress Loop Tour",
                        "ticket_cost": 60,
                    },
                    {
                        "activity_id": 303,
                        "city_id": 103,
                        "activity_name": "Harbor Sunset Ferry",
                        "ticket_cost": 75,
                    },
                ],
            },
        ),
        ScriptedAtomicToolCall(
            tool_name="neighborhood",
            params={"table": "proof_cities", "row_id": 101, "max_per_edge": 5},
            result={
                "anchor": {
                    "table": "proof_cities",
                    "row_id": 101,
                    "attributes": {"city_id": 101, "city_name": "Seoul"},
                },
                "edges": [
                    {
                        "destination_table": "proof_city_links",
                        "total_count": 1,
                        "sample_ids": [102],
                        "preview": [{"neighbor_city_id": 102, "city_name": "Suwon"}],
                    }
                ],
            },
        ),
        ScriptedAtomicToolCall(
            tool_name="query",
            params={"spec": {"proof": "canonical_itinerary"}},
            result={
                "columns": [
                    "day",
                    "city",
                    "lodging",
                    "activity",
                    "total_cost",
                ],
                "column_sources": [
                    {
                        "output": "day",
                        "kind": "select",
                        "table": "proof_itinerary",
                        "column": "day",
                        "visibility": "user_visible",
                        "is_handle": False,
                        "value_exposes_source": True,
                    },
                    {
                        "output": "city",
                        "kind": "select",
                        "table": "proof_cities",
                        "column": "city_name",
                        "visibility": "user_visible",
                        "is_handle": False,
                        "value_exposes_source": True,
                    },
                    {
                        "output": "lodging",
                        "kind": "select",
                        "table": "proof_lodgings",
                        "column": "lodging_name",
                        "visibility": "user_visible",
                        "is_handle": False,
                        "value_exposes_source": True,
                    },
                    {
                        "output": "activity",
                        "kind": "select",
                        "table": "proof_activities",
                        "column": "activity_name",
                        "visibility": "user_visible",
                        "is_handle": False,
                        "value_exposes_source": True,
                    },
                    {
                        "output": "total_cost",
                        "kind": "select",
                        "table": "proof_itinerary",
                        "column": "total_cost",
                        "visibility": "user_visible",
                        "is_handle": False,
                        "value_exposes_source": True,
                    },
                ],
                "referenced_columns": [
                    {
                        "usage": "where",
                        "table": "proof_itinerary",
                        "column": "total_cost",
                        "visibility": "user_visible",
                        "is_handle": False,
                        "op": "lte",
                        "value": 250,
                    },
                    {
                        "usage": "order_by",
                        "table": "proof_cities",
                        "column": "city_name",
                        "visibility": "user_visible",
                        "is_handle": False,
                        "direction": "asc",
                    },
                ],
                "rows": PROOF_CANONICAL_ANSWER,
                "row_count": len(PROOF_CANONICAL_ANSWER),
            },
        ),
    ]
    submit_payload = SubmitDraftPayload.model_validate(
        {
            "topic": PROOF_TASK_TOPIC,
            "label_json": PROOF_CANONICAL_ANSWER,
            "entity_json": PROOF_ANCHOR_ENTITY,
            "user_request": build_proof_question(),
            "answer_contract": {
                "kind": "list",
                "answer_phrase": "일정표",
                "constraint_phrases": [
                    "total_cost는 250 이하여야",
                    "사전순이 가장 앞서는",
                ],
                "limit_phrase": "3일",
                "output_bindings": [
                    {
                        "label_field": "day",
                        "requested_by_phrase": "day",
                    },
                    {
                        "label_field": "city",
                        "requested_by_phrase": "city",
                    },
                    {
                        "label_field": "lodging",
                        "requested_by_phrase": "lodging",
                    },
                    {
                        "label_field": "activity",
                        "requested_by_phrase": "activity",
                    },
                    {
                        "label_field": "total_cost",
                        "requested_by_phrase": "total_cost",
                    },
                ],
                "order_bindings": [
                    {
                        "direction": "asc",
                        "label_field": "city",
                        "requested_by_phrase": "사전순이 가장 앞서는",
                    }
                ],
            },
        }
    )
    return ScriptedComposerScript(
        atomic_tool_calls=tuple(atomic_tool_calls),
        submit_payload=submit_payload,
        final_output_text="proof scripted composer",
        turn_count=len(atomic_tool_calls) + 1,
    )


@dataclass(slots=True)
class _AlternatingProofSolverRuntime:
    """Solver runtime that alternates canonical/non-canonical answers.

    Shared across all solver rollouts in one proof run so the matched-count
    ratio is deterministic (5/10, 4/8, etc.) and lands inside the quality-gate
    band. The instance is stateful, which is acceptable because the proof
    pipeline is single-threaded by design.
    """

    canonical_answer_json: str
    unmatched_answer_json: str
    solver_id: str = "proof_scripted"
    provider_name: str = "scripted"
    model_name: str = "scripted"
    _counter: int = field(default=0, init=False, repr=False)

    async def run(self, episode: SolverEpisodeInput) -> SolverResult:
        matched = self._counter % 2 == 0
        self._counter += 1
        raw_output = (
            self.canonical_answer_json if matched else self.unmatched_answer_json
        )
        return SolverResult(
            task_id=episode.task_id,
            solver_id=self.solver_id,
            provider=self.provider_name,
            model=self.model_name,
            raw_output_text=raw_output,
            structured_output=None,
            status="completed",
            termination_reason="submitted",
        )


def _build_proof_runtime_factory(
    shared: _AlternatingProofSolverRuntime,
) -> object:
    def _factory(*_args: object, **_kwargs: object) -> AgentRuntime:
        return shared

    return _factory


async def _empty_proof_sdk_tools(_task_bundle: TaskBundleContract) -> list[object]:
    return []


def _proof_provider_name(config: AppConfig) -> str:
    if not config.providers:
        raise RuntimeError(
            "proof smoke fixture requires at least one configured provider "
            "in AppConfig.providers"
        )
    return next(iter(config.providers))


def _build_proof_app_config(base_config: AppConfig, schema_name: str) -> AppConfig:
    database_override = base_config.database.model_copy(
        update={"schema_allowlist": [schema_name]}
    )
    calibration_override = base_config.calibration.model_copy(
        update={
            "max_solver_runs": 6,
            "solver_batch_size": 3,
            "safe_early_termination": False,
        }
    )
    return base_config.model_copy(
        update={
            "database": database_override,
            "calibration": calibration_override,
        },
        deep=True,
    )


async def _ensure_proof_schema(database: DatabaseConfig, schema_name: str) -> None:
    quoted = f'"{schema_name}"'
    conn = await asyncpg.connect(dsn=database.dsn)
    try:
        await _apply_session_settings(
            conn,
            mutating_control_session_settings(database),
        )
        async with conn.transaction():
            await conn.execute(f"CREATE SCHEMA {quoted}")
            await conn.execute(f"SET LOCAL search_path TO {quoted}")
            for statement in _split_sql_statements(PROOF_FIXTURE_DDL):
                await conn.execute(statement)
            for statement in _split_sql_statements(PROOF_FIXTURE_SEED):
                await conn.execute(statement)
            if database.readonly_role:
                role = quote_ident(database.readonly_role)
                await conn.execute(f"GRANT USAGE ON SCHEMA {quoted} TO {role}")
                await conn.execute(
                    f"GRANT SELECT ON ALL TABLES IN SCHEMA {quoted} TO {role}"
                )
    finally:
        await conn.close()


async def _drop_proof_schema(database: DatabaseConfig, schema_name: str) -> None:
    quoted = f'"{schema_name}"'
    conn = await asyncpg.connect(dsn=database.dsn)
    try:
        await _apply_session_settings(
            conn,
            mutating_control_session_settings(database),
        )
        await conn.execute(f"DROP SCHEMA IF EXISTS {quoted} CASCADE")
    finally:
        await conn.close()


async def run_proof_task(
    config: AppConfig,
    *,
    output_root: Path,
    mirror_analysis_log_path: Path | None = None,
    schema_name: str | None = None,
) -> RealDbTrialSummary:
    """Provision an ephemeral proof schema and drive a trial through the runtime."""

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_schema = schema_name or f"proof_trial_{uuid.uuid4().hex[:8]}"
    proof_config = _build_proof_app_config(config, resolved_schema)
    pools = await DatabasePools.create(proof_config.database)
    await _ensure_proof_schema(proof_config.database, resolved_schema)
    synthesis_db: SynthesisDb | None = None
    solver_orchestrator: SolverOrchestrator | None = None
    try:
        synthesis_db = SynthesisDb(
            db_id=PROOF_DB_ID,
            config=proof_config,
            database_pools=pools,
        )
        synthesis_db.adopt_schema_graph(build_proof_schema_graph(resolved_schema))
        proof_runtime = _AlternatingProofSolverRuntime(
            canonical_answer_json=_canonical_answer_json(),
            unmatched_answer_json=_unmatched_answer_json(),
        )
        solver_orchestrator = SolverOrchestrator(
            proof_config,
            database_pools=pools,
            runtime_factory=_build_proof_runtime_factory(proof_runtime),
            sdk_tools_factory=_empty_proof_sdk_tools,
        )
        provider_name = _proof_provider_name(proof_config)
        scripted_backend = ScriptedComposerBackend(
            script=build_proof_composer_script(),
            provider_name=provider_name,
            model_name="scripted-proof",
        )
        runner = RealDbTrialRunner(
            proof_config,
            database_pools=pools,
            solver_orchestrator=solver_orchestrator,
            synthesis_db=synthesis_db,
            synthesis_backends=[scripted_backend],
        )
        try:
            summary = await runner.run(
                output_root,
                db_id=PROOF_DB_ID,
                topic=PROOF_TASK_TOPIC,
                mirror_analysis_log_path=mirror_analysis_log_path,
            )
        finally:
            await runner.close()
        return summary
    finally:
        try:
            if solver_orchestrator is not None:
                await solver_orchestrator.close()
            if synthesis_db is not None:
                await synthesis_db.close()
        finally:
            try:
                await _drop_proof_schema(proof_config.database, resolved_schema)
            finally:
                await pools.close()


__all__ = [
    "PROOF_ANCHOR_ENTITY",
    "PROOF_CANONICAL_ANSWER",
    "PROOF_DB_ID",
    "PROOF_FIXTURE_DDL",
    "PROOF_FIXTURE_SEED",
    "PROOF_QUESTION_BODY",
    "PROOF_TASK_TOPIC",
    "build_proof_composer_script",
    "build_proof_question",
    "build_proof_schema_graph",
    "run_proof_task",
]
