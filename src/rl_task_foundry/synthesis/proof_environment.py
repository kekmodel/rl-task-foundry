"""Synthetic proof-task vertical slice helpers.

This module keeps the first Milestone 5 proof task deterministic and reviewable:
it defines a synthetic fixture DB schema, a compositional itinerary environment,
and a small runner that executes the same rollout -> quality gate -> registry ->
bundle export path used by accepted synthesized task bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.pipeline.solver_orchestrator import (
    SolverOrchestrator,
    TaskQualityGateStatus,
    evaluate_rollout_summary,
)
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolBundle,
    AtomicToolDefinition,
    AtomicToolFamily,
    AtomicToolResultMode,
)
from rl_task_foundry.synthesis.backend_openai_agents import OpenAIAgentsSynthesisBackend
from rl_task_foundry.synthesis.bundle_exporter import TaskBundleExporter
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import (
    ConstraintKind,
    ConstraintSummaryItem,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    TaskBundleContract,
    TaskBundleStatus,
    TaskContract,
    TaskQualityMetrics,
)
from rl_task_foundry.synthesis.phase_monitor import (
    PipelinePhaseMonitorLogger,
)
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics
from rl_task_foundry.synthesis.rendered_prompt_builder import build_rendered_user_prompt
from rl_task_foundry.synthesis.runtime import (
    CURRENT_SYNTHESIS_GENERATOR_VERSION,
    SynthesisTaskDraft,
)
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryCommitStatus,
    TaskRegistryWriter,
)
from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.tooling.common import SchemaSnapshot, snapshot_from_graph

PROOF_DB_ID = "proof_trip_fixture"
PROOF_TASK_ID = "task_proof_trip_fixture_itinerary_v1"

PROOF_FIXTURE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS proof_anchors (
    anchor_id INTEGER PRIMARY KEY,
    season TEXT NOT NULL,
    budget_bucket TEXT NOT NULL,
    start_city_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS proof_cities (
    city_id INTEGER PRIMARY KEY,
    city_name TEXT NOT NULL,
    region_name TEXT NOT NULL,
    season TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proof_city_links (
    city_id INTEGER NOT NULL,
    neighbor_city_id INTEGER NOT NULL,
    PRIMARY KEY (city_id, neighbor_city_id),
    FOREIGN KEY (city_id) REFERENCES proof_cities(city_id),
    FOREIGN KEY (neighbor_city_id) REFERENCES proof_cities(city_id)
);

CREATE TABLE IF NOT EXISTS proof_lodgings (
    lodging_id INTEGER PRIMARY KEY,
    city_id INTEGER NOT NULL,
    lodging_name TEXT NOT NULL,
    nightly_cost INTEGER NOT NULL,
    FOREIGN KEY (city_id) REFERENCES proof_cities(city_id)
);

CREATE TABLE IF NOT EXISTS proof_activities (
    activity_id INTEGER PRIMARY KEY,
    city_id INTEGER NOT NULL,
    activity_name TEXT NOT NULL,
    ticket_cost INTEGER NOT NULL,
    FOREIGN KEY (city_id) REFERENCES proof_cities(city_id)
);
""".strip()

PROOF_FIXTURE_SEED_SQL = """
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


@dataclass(frozen=True, slots=True)
class ProofFixtureSqlFiles:
    root_dir: Path
    schema_path: Path
    seed_path: Path


@dataclass(frozen=True, slots=True)
class ProofTaskRunSummary:
    db_id: str
    task_id: str
    fixture_sql_root: Path
    quality_gate_status: str
    flow_id: str | None = None
    phase_monitor_log_path: Path | None = None
    solver_pass_rate: float | None = None
    solver_ci_low: float | None = None
    solver_ci_high: float | None = None
    registry_status: TaskRegistryCommitStatus | None = None
    registry_task_id: str | None = None
    bundle_root: Path | None = None


@dataclass(slots=True)
class ProofTaskRunner:
    config: AppConfig
    solver_orchestrator: SolverOrchestrator | None = None
    registry: TaskRegistryWriter | None = None
    exporter: TaskBundleExporter | None = None

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = TaskRegistryWriter.for_config(self.config)
        assert self.registry.snapshot_materializer is not None
        if self.exporter is None:
            self.exporter = TaskBundleExporter(
                registry=self.registry,
                snapshot_materializer=self.registry.snapshot_materializer,
            )
        if self.solver_orchestrator is None:
            self.solver_orchestrator = SolverOrchestrator(self.config)

    async def run(self, output_root: Path) -> ProofTaskRunSummary:
        output_root.mkdir(parents=True, exist_ok=True)
        assert self.registry is not None
        assert self.registry.snapshot_materializer is not None
        self.registry.snapshot_materializer.materialize(
            db_id=PROOF_DB_ID,
            snapshot=build_proof_schema_snapshot(),
        )
        flow_id = build_flow_id("proof_task")
        phase_monitor_log_path = output_root / "debug" / "phase_monitors.jsonl"
        phase_monitor = PipelinePhaseMonitorLogger(
            phase_monitor_log_path=phase_monitor_log_path,
            flow_kind="proof_task",
            flow_id=flow_id,
        )
        try:
            fixture_files = write_proof_fixture_sql(output_root / "fixture_db")
            draft = build_proof_task_draft(config=self.config)
            phase_monitor.emit(
                phase="draft_build",
                status="completed",
                expected_contract={
                    "db_id": PROOF_DB_ID,
                    "topic": "itinerary",
                },
                actual_data={
                    "task_id": draft.task_bundle.task_id,
                    "rendered_user_prompt": draft.rendered_user_prompt,
                    "canonical_answer_json": draft.canonical_answer_json,
                },
                checks={"label_signature_present": bool(draft.label_signature)},
                diagnostics={},
            )
            assert self.solver_orchestrator is not None
            rollout_summary = await self.solver_orchestrator.run_draft(draft)
            phase_monitor.emit(
                phase="rollout",
                status="completed",
                expected_contract={
                    "max_solver_runs": self.config.calibration.max_solver_runs,
                },
                actual_data={
                    "planned_solver_runs": rollout_summary.planned_solver_runs,
                    "total_solver_runs": rollout_summary.total_solver_runs,
                    "matched_solver_runs": rollout_summary.matched_solver_runs,
                    "early_stop_decision": rollout_summary.early_stop_decision,
                },
                checks={
                    "executed_runs_within_plan": rollout_summary.total_solver_runs
                    <= rollout_summary.planned_solver_runs,
                },
                diagnostics={"task_id": draft.task_bundle.task_id},
            )
            quality_gate_summary = evaluate_rollout_summary(self.config, rollout_summary)
            phase_monitor.emit(
                phase="quality_gate",
                status=quality_gate_summary.status.value,
                expected_contract={
                    "band_lower": quality_gate_summary.band_lower,
                    "band_upper": quality_gate_summary.band_upper,
                },
                actual_data={
                    "pass_rate": quality_gate_summary.pass_rate,
                    "ci_low": quality_gate_summary.ci_lower,
                    "ci_high": quality_gate_summary.ci_upper,
                },
                checks={
                    "accepted": quality_gate_summary.status is TaskQualityGateStatus.ACCEPT,
                },
                diagnostics={"task_id": draft.task_bundle.task_id},
            )
            if quality_gate_summary.status is not TaskQualityGateStatus.ACCEPT:
                return ProofTaskRunSummary(
                    db_id=draft.task_bundle.db_id,
                    task_id=draft.task_bundle.task_id,
                    fixture_sql_root=fixture_files.root_dir,
                    quality_gate_status=quality_gate_summary.status.value,
                    flow_id=flow_id,
                    phase_monitor_log_path=phase_monitor_log_path,
                    solver_pass_rate=quality_gate_summary.pass_rate,
                    solver_ci_low=quality_gate_summary.ci_lower,
                    solver_ci_high=quality_gate_summary.ci_upper,
                )

            accepted_draft = accepted_draft_with_quality_metrics(
                draft,
                quality_gate_summary=quality_gate_summary,
            )
            assert self.registry is not None
            commit_result = self.registry.commit_draft(accepted_draft)
            phase_monitor.emit(
                phase="registry_commit",
                status=commit_result.status.value,
                expected_contract={},
                actual_data={
                    "registry_task_id": commit_result.task_id,
                    "status": commit_result.status.value,
                },
                checks={},
                diagnostics={"task_id": accepted_draft.task_bundle.task_id},
            )
            assert self.exporter is not None
            bundle_root = output_root / "bundle"
            self.exporter.export_bundle(bundle_root, task_id=commit_result.task_id)
            phase_monitor.emit(
                phase="bundle_export",
                status="completed",
                expected_contract={"bundle_root": bundle_root},
                actual_data={"bundle_root": bundle_root, "task_id": commit_result.task_id},
                checks={"bundle_root_exists": bundle_root.exists()},
                diagnostics={},
            )
            return ProofTaskRunSummary(
                db_id=accepted_draft.task_bundle.db_id,
                task_id=accepted_draft.task_bundle.task_id,
                fixture_sql_root=fixture_files.root_dir,
                quality_gate_status=quality_gate_summary.status.value,
                flow_id=flow_id,
                phase_monitor_log_path=phase_monitor_log_path,
                solver_pass_rate=quality_gate_summary.pass_rate,
                solver_ci_low=quality_gate_summary.ci_lower,
                solver_ci_high=quality_gate_summary.ci_upper,
                registry_status=commit_result.status,
                registry_task_id=commit_result.task_id,
                bundle_root=bundle_root,
            )
        finally:
            phase_monitor.close()

    async def close(self) -> None:
        assert self.solver_orchestrator is not None
        await self.solver_orchestrator.close()
        OpenAIAgentsSynthesisBackend.clear_model_cache()
        close_registry = getattr(self.registry, "close", None)
        if callable(close_registry):
            close_registry()


def build_proof_schema_snapshot() -> "SchemaSnapshot":
    """Hand-written snapshot matching `PROOF_FIXTURE_SCHEMA_SQL`.

    Lives next to the fixture SQL so both paths stay in sync. The proof
    harness feeds this into `SchemaSnapshotMaterializer` before bundle
    export so the exported proof bundle carries a valid
    `schema_snapshot.json` without round-tripping through
    introspection.
    """
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="proof_anchors",
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_anchors",
                        column_name="anchor_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_anchors",
                        column_name="season",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_anchors",
                        column_name="budget_bucket",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
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
                schema_name="public",
                table_name="proof_cities",
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_cities",
                        column_name="city_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_cities",
                        column_name="city_name",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_cities",
                        column_name="region_name",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
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
                schema_name="public",
                table_name="proof_lodgings",
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_lodgings",
                        column_name="lodging_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_lodgings",
                        column_name="city_id",
                        data_type="integer",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_lodgings",
                        column_name="lodging_name",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
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
                schema_name="public",
                table_name="proof_activities",
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_activities",
                        column_name="activity_id",
                        data_type="integer",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="user_visible",
                        is_primary_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_activities",
                        column_name="city_id",
                        data_type="integer",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="proof_activities",
                        column_name="activity_name",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
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
                source_schema="public",
                source_table="proof_anchors",
                source_columns=("start_city_id",),
                target_schema="public",
                target_table="proof_cities",
                target_columns=("city_id",),
            ),
            ForeignKeyEdge(
                constraint_name="lodgings_city",
                source_schema="public",
                source_table="proof_lodgings",
                source_columns=("city_id",),
                target_schema="public",
                target_table="proof_cities",
                target_columns=("city_id",),
            ),
            ForeignKeyEdge(
                constraint_name="activities_city",
                source_schema="public",
                source_table="proof_activities",
                source_columns=("city_id",),
                target_schema="public",
                target_table="proof_cities",
                target_columns=("city_id",),
            ),
        ],
    )
    return snapshot_from_graph(graph)


def write_proof_fixture_sql(root_dir: Path) -> ProofFixtureSqlFiles:
    root_dir.mkdir(parents=True, exist_ok=True)
    schema_path = root_dir / "schema.sql"
    seed_path = root_dir / "seed.sql"
    schema_path.write_text(PROOF_FIXTURE_SCHEMA_SQL + "\n", encoding="utf-8")
    seed_path.write_text(PROOF_FIXTURE_SEED_SQL + "\n", encoding="utf-8")
    return ProofFixtureSqlFiles(
        root_dir=root_dir,
        schema_path=schema_path,
        seed_path=seed_path,
    )


def build_proof_task_draft(
    *,
    config: AppConfig,
    created_at: datetime | None = None,
) -> SynthesisTaskDraft:
    created_at = created_at or datetime.now(timezone.utc)
    output_schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="itinerary",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("day",),
            unique_elements=True,
            items=OutputFieldContract(
                name="day_plan",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="day", type=OutputFieldType.INT),
                    OutputFieldContract(name="city", type=OutputFieldType.STRING),
                    OutputFieldContract(name="lodging", type=OutputFieldType.STRING),
                    OutputFieldContract(name="activity", type=OutputFieldType.STRING),
                    OutputFieldContract(name="total_cost", type=OutputFieldType.INT),
                ],
            ),
        ),
        primary_output_format="json_array",
    )
    canonical_answer = [
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
    canonical_answer_json = canonical_json(canonical_answer)
    label_signature = "sha256:" + sha256(canonical_answer_json.encode("utf-8")).hexdigest()

    task = TaskContract(
        question=(
            "봄 시즌 3일 출장 일정표를 만들어 주세요. 각 day는 하나의 city를 방문해야 하고 "
            "전체 itinerary에서 city는 중복되면 안 됩니다. day별 total_cost는 250 이하여야 "
            "하며, 연속된 day의 city는 인접한 지역이어야 합니다. 가능한 일정이 여러 개면 "
            "day 1의 city 이름, 그다음 day 2의 city 이름, 그다음 day 3의 city 이름의 "
            "사전순이 가장 앞서는 답을 고르세요."
        ),
        topic="itinerary",
        output_schema=output_schema,
        constraint_summary=[
            ConstraintSummaryItem(
                key="three_days",
                kind=ConstraintKind.CARDINALITY,
                summary="day 1, day 2, day 3 세 슬롯을 모두 채워야 한다.",
            ),
            ConstraintSummaryItem(
                key="unique_city",
                kind=ConstraintKind.UNIQUENESS,
                summary="전체 일정에서 city는 중복되면 안 된다.",
            ),
            ConstraintSummaryItem(
                key="daily_budget",
                kind=ConstraintKind.RANGE,
                summary="각 day의 total_cost는 250 이하여야 한다.",
            ),
            ConstraintSummaryItem(
                key="adjacent_cities",
                kind=ConstraintKind.TEMPORAL,
                summary="연속된 day의 city는 proof_city_links 기준으로 인접해야 한다.",
            ),
        ],
        instance_parameters={
            "anchor_id": 1,
            "season": "spring",
            "budget_bucket": "mid",
        },
    )
    task_bundle = TaskBundleContract(
        task_id=PROOF_TASK_ID,
        db_id=PROOF_DB_ID,
        domain="travel_planning",
        topic="itinerary",
        atomic_tool_set_ref=f"db://{PROOF_DB_ID}",
        created_at=created_at,
        generator_version=CURRENT_SYNTHESIS_GENERATOR_VERSION,
        tool_signature=_sha256_hex(_proof_atomic_tool_bundle().source),
        task_signature=_sha256_hex(task.model_dump_json()),
        status=TaskBundleStatus.DRAFT,
        quality_metrics=TaskQualityMetrics(),
        rollout_constraints=RolloutConstraintsContract(
            max_turns=config.solver_runtime.max_turns,
            max_episode_duration_ms=(
                config.database.statement_timeout_ms * config.solver_runtime.max_turns
            ),
            max_tool_rows=config.atomic_tools.bounded_result_limit,
        ),
        task=task,
    )
    anchor_entity: dict[str, object] = {"anchor_id": 1}
    rendered_prompt = build_rendered_user_prompt(
        task,
        anchor_entity=anchor_entity,
        canonical_answer=canonical_answer,
    )
    return SynthesisTaskDraft(
        created_at=created_at,
        db_id=PROOF_DB_ID,
        requested_topic="itinerary",
        schema_summary={"included_table_count": 5, "fixture": "proof_trip_fixture"},
        selected_topic="itinerary",
        task_bundle=task_bundle,
        atomic_tool_bundle=_proof_atomic_tool_bundle(),
        rendered_user_prompt=rendered_prompt,
        anchor_entity=anchor_entity,
        canonical_answer_json=canonical_answer_json,
        label_signature=label_signature,
        generation_attempts=[],
        provider_status={},
    )


def _proof_atomic_tool_bundle() -> AtomicToolBundle:
    tools = [
        AtomicToolDefinition(
            name="get_proof_anchor",
            family=AtomicToolFamily.GET,
            description="Retrieve one proof anchor by ID. Returns all fields or nothing.",
            params_schema={
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            returns_schema={
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "anchor_id": {"type": "integer"},
                            "season": {"type": "string"},
                            "budget_bucket": {"type": "string"},
                            "start_city_id": {"type": "integer"},
                        },
                        "required": ["anchor_id", "season", "budget_bucket", "start_city_id"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
            sql=(
                "SELECT anchor_id, season, budget_bucket, start_city_id "
                "FROM proof_anchors WHERE anchor_id = $1 LIMIT 1"
            ),
            result_mode=AtomicToolResultMode.OBJECT_OR_NULL,
            semantic_key="proof_anchors:get",
        ),
        AtomicToolDefinition(
            name="find_proof_city_by_season",
            family=AtomicToolFamily.FIND,
            description="Find proof city entries where season matches a condition. Returns a list.",
            params_schema={
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["any", "eq", "in", "like"]},
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 20,
                            },
                            {"type": "null"},
                        ]
                    },
                    "sort_by": {
                        "anyOf": [
                            {"type": "string", "enum": ["city_id", "city_name", "region_name"]},
                            {"type": "null"},
                        ]
                    },
                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["op", "value", "sort_by", "direction", "limit"],
                "additionalProperties": False,
            },
            returns_schema={
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "city_id": {"type": "integer"},
                        "city_name": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["city_id", "city_name", "region_name"],
                },
            },
            sql=(
                "SELECT city_id, city_name, region_name "
                "FROM proof_cities WHERE TRUE ORDER BY city_id LIMIT $1"
            ),
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key="proof_cities:find:season",
        ),
        AtomicToolDefinition(
            name="find_proof_city_link_by_city_id",
            family=AtomicToolFamily.FIND,
            description="Find proof city link entries where city id matches a condition. Returns a list.",  # noqa: E501
            params_schema={
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["any", "eq", "in", "lt", "gt", "lte", "gte"]},
                    "value": {
                        "anyOf": [
                            {"type": "integer"},
                            {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 1,
                                "maxItems": 20,
                            },
                            {"type": "null"},
                        ]
                    },
                    "sort_by": {
                        "anyOf": [
                            {"type": "string", "enum": ["neighbor_city_id"]},
                            {"type": "null"},
                        ]
                    },
                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["op", "value", "sort_by", "direction", "limit"],
                "additionalProperties": False,
            },
            returns_schema={
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {"neighbor_city_id": {"type": "integer"}},
                    "required": ["neighbor_city_id"],
                },
            },
            sql=(
                "SELECT neighbor_city_id FROM proof_city_links WHERE TRUE ORDER BY neighbor_city_id LIMIT $1"  # noqa: E501
            ),
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key="proof_city_links:find:city_id",
        ),
        AtomicToolDefinition(
            name="find_proof_lodging_by_city_id",
            family=AtomicToolFamily.FIND,
            description="Find proof lodging entries where city id matches a condition. Returns a list.",  # noqa: E501
            params_schema={
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["any", "eq", "in", "lt", "gt", "lte", "gte"]},
                    "value": {
                        "anyOf": [
                            {"type": "integer"},
                            {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 1,
                                "maxItems": 20,
                            },
                            {"type": "null"},
                        ]
                    },
                    "sort_by": {
                        "anyOf": [
                            {
                                "type": "string",
                                "enum": ["lodging_id", "lodging_name", "nightly_cost"],
                            },
                            {"type": "null"},
                        ]
                    },
                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["op", "value", "sort_by", "direction", "limit"],
                "additionalProperties": False,
            },
            returns_schema={
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "lodging_id": {"type": "integer"},
                        "lodging_name": {"type": "string"},
                        "nightly_cost": {"type": "integer"},
                    },
                    "required": ["lodging_id", "lodging_name", "nightly_cost"],
                },
            },
            sql=(
                "SELECT lodging_id, lodging_name, nightly_cost FROM proof_lodgings WHERE TRUE ORDER BY lodging_id LIMIT $1"  # noqa: E501
            ),
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key="proof_lodgings:find:city_id",
        ),
        AtomicToolDefinition(
            name="find_proof_activity_by_city_id",
            family=AtomicToolFamily.FIND,
            description="Find proof activity entries where city id matches a condition. Returns a list.",  # noqa: E501
            params_schema={
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["any", "eq", "in", "lt", "gt", "lte", "gte"]},
                    "value": {
                        "anyOf": [
                            {"type": "integer"},
                            {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 1,
                                "maxItems": 20,
                            },
                            {"type": "null"},
                        ]
                    },
                    "sort_by": {
                        "anyOf": [
                            {
                                "type": "string",
                                "enum": ["activity_id", "activity_name", "ticket_cost"],
                            },
                            {"type": "null"},
                        ]
                    },
                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["op", "value", "sort_by", "direction", "limit"],
                "additionalProperties": False,
            },
            returns_schema={
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "activity_id": {"type": "integer"},
                        "activity_name": {"type": "string"},
                        "ticket_cost": {"type": "integer"},
                    },
                    "required": ["activity_id", "activity_name", "ticket_cost"],
                },
            },
            sql=(
                "SELECT activity_id, activity_name, ticket_cost FROM proof_activities WHERE TRUE ORDER BY activity_id LIMIT $1"  # noqa: E501
            ),
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key="proof_activities:find:city_id",
        ),
    ]
    source = (
        """
\"\"\"Atomic tools for the proof_trip_fixture database.\"\"\"
from __future__ import annotations


async def get_proof_anchor(conn, id):
    row = await conn.fetchrow(
        \"\"\"
        SELECT anchor_id, season, budget_bucket, start_city_id
        FROM proof_anchors
        WHERE anchor_id = $1
        LIMIT 1
        \"\"\",
        id,
    )
    return None if row is None else dict(row)


async def find_proof_city_by_season(conn, op, value, sort_by, direction, limit, _shuffle_seed=None):
    limit = min(limit, 20)
    if op not in {\"any\", \"eq\"}:
        raise ValueError(\"unsupported op\")
    if op == \"any\":
        rows = await conn.fetch(
            \"\"\"
            SELECT city_id, city_name, region_name
            FROM proof_cities
            ORDER BY city_id
            LIMIT $1
            \"\"\",
            limit,
        )
    else:
        rows = await conn.fetch(
            \"\"\"
            SELECT city_id, city_name, region_name
            FROM proof_cities
            WHERE season = $1
            ORDER BY city_id
            LIMIT $2
            \"\"\",
            value,
            limit,
        )
    return [dict(row) for row in rows]


async def find_proof_city_link_by_city_id(
    conn, op, value, sort_by, direction, limit, _shuffle_seed=None,
):
    limit = min(limit, 20)
    rows = await conn.fetch(
        \"\"\"
        SELECT neighbor_city_id
        FROM proof_city_links
        WHERE city_id = $1
        ORDER BY neighbor_city_id
        LIMIT $2
        \"\"\",
        value,
        limit,
    )
    return [dict(row) for row in rows]


async def find_proof_lodging_by_city_id(
    conn, op, value, sort_by, direction, limit, _shuffle_seed=None,
):
    limit = min(limit, 20)
    rows = await conn.fetch(
        \"\"\"
        SELECT lodging_id, lodging_name, nightly_cost
        FROM proof_lodgings
        WHERE city_id = $1
        ORDER BY lodging_id
        LIMIT $2
        \"\"\",
        value,
        limit,
    )
    return [dict(row) for row in rows]


async def find_proof_activity_by_city_id(
    conn, op, value, sort_by, direction, limit, _shuffle_seed=None,
):
    limit = min(limit, 20)
    rows = await conn.fetch(
        \"\"\"
        SELECT activity_id, activity_name, ticket_cost
        FROM proof_activities
        WHERE city_id = $1
        ORDER BY activity_id
        LIMIT $2
        \"\"\",
        value,
        limit,
    )
    return [dict(row) for row in rows]
""".strip()
        + "\n"
    )
    return AtomicToolBundle(db_id=PROOF_DB_ID, tools=tools, source=source)


def _sha256_hex(payload: str) -> str:
    return "sha256:" + sha256(payload.encode("utf-8")).hexdigest()
