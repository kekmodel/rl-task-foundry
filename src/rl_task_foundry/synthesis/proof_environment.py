"""Synthetic proof-environment vertical slice helpers.

This module keeps the first Milestone 5 proof task deterministic and reviewable:
it defines a synthetic fixture DB schema, a compositional itinerary environment,
and a small runner that executes the same rollout -> quality gate -> registry ->
bundle export path used by accepted synthesized environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.pipeline.environment_orchestrator import (
    EnvironmentOrchestrator,
    EnvironmentQualityGateStatus,
    evaluate_rollout_summary,
)
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolBundle,
    AtomicToolDefinition,
    AtomicToolFamily,
    AtomicToolResultMode,
)
from rl_task_foundry.synthesis.bundle_exporter import EnvironmentBundleExporter
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import (
    AnchorQueryContract,
    CategoryTaxonomy,
    ConstraintKind,
    ConstraintSummaryItem,
    CrossInstanceSet,
    DifficultyVectorContract,
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnvironmentStatus,
    InstanceContract,
    InstanceSpaceContract,
    MaterializedFactsSchema,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    ShadowVerifierContract,
    SolutionContract,
    TaskContract,
    VerifierContract,
    difficulty_vector_json,
)
from rl_task_foundry.synthesis.cross_instance import evaluate_cross_instance_draft
from rl_task_foundry.synthesis.environment_registry import (
    EnvironmentRegistryCommitStatus,
    EnvironmentRegistryWriter,
)
from rl_task_foundry.synthesis.phase_monitor import (
    PipelinePhaseMonitorLogger,
)
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
from rl_task_foundry.synthesis.rendered_prompt_builder import build_rendered_user_prompt
from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics
from rl_task_foundry.synthesis.registration_policy import ArtifactKind
from rl_task_foundry.synthesis.registration_runner import (
    ArtifactRegistrationResult,
    GeneratedArtifactBundle,
    RegistrationArtifactName,
    RegistrationBundleDiagnostics,
    RegistrationBundleReport,
    RegistrationBundleStatus,
    build_registration_diagnostics,
)
from rl_task_foundry.synthesis.runtime import (
    CURRENT_SYNTHESIS_GENERATOR_VERSION,
    MaterializedCanonicalAnswerRecord,
    MaterializedInstanceRecord,
    SynthesisEnvironmentDraft,
    SynthesisSelfConsistencyDiagnostics,
)

PROOF_DB_ID = "proof_trip_fixture"
PROOF_ENV_ID = "env_proof_trip_fixture_itinerary_v1"

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
class ProofEnvironmentRunSummary:
    db_id: str
    env_id: str
    fixture_sql_root: Path
    quality_gate_status: str
    flow_id: str | None = None
    phase_monitor_log_path: Path | None = None
    solver_pass_rate: float | None = None
    solver_ci_low: float | None = None
    solver_ci_high: float | None = None
    cross_instance_error_codes: tuple[str, ...] = ()
    registry_status: EnvironmentRegistryCommitStatus | None = None
    registry_env_id: str | None = None
    bundle_root: Path | None = None


@dataclass(slots=True)
class ProofEnvironmentRunner:
    config: AppConfig
    environment_orchestrator: EnvironmentOrchestrator | None = None
    registry: EnvironmentRegistryWriter | None = None
    exporter: EnvironmentBundleExporter | None = None

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = EnvironmentRegistryWriter.for_config(self.config)
        if self.exporter is None:
            self.exporter = EnvironmentBundleExporter(
                registry=self.registry,
                materializer=self.registry.atomic_tool_materializer,
            )
        if self.environment_orchestrator is None:
            self.environment_orchestrator = EnvironmentOrchestrator(self.config)

    async def run(self, output_root: Path) -> ProofEnvironmentRunSummary:
        output_root.mkdir(parents=True, exist_ok=True)
        flow_id = build_flow_id("proof_environment")
        phase_monitor_log_path = output_root / "debug" / "phase_monitors.jsonl"
        phase_monitor = PipelinePhaseMonitorLogger(
            phase_monitor_log_path=phase_monitor_log_path,
            flow_kind="proof_environment",
            flow_id=flow_id,
        )
        fixture_files = write_proof_fixture_sql(output_root / "fixture_db")
        draft = build_proof_environment_draft()
        phase_monitor.emit(
            phase="draft_build",
            status="completed",
            expected_contract={
                "db_id": PROOF_DB_ID,
                "category": CategoryTaxonomy.ITINERARY.value,
            },
            actual_data={
                "env_id": draft.environment.env_id,
                "difficulty_vector": difficulty_vector_json(
                    draft.environment.task.difficulty_vector
                ),
                "rendered_user_prompts": [
                    instance.rendered_user_prompt for instance in draft.instances
                ],
                "canonical_answer_jsons": [
                    answer.canonical_answer_json for answer in draft.canonical_answers
                ],
            },
            checks={
                "instance_count_matches_canonical_answers": len(draft.instances)
                == len(draft.canonical_answers),
            },
            diagnostics={},
        )
        cross_instance_summary = evaluate_cross_instance_draft(draft)
        phase_monitor.emit(
            phase="cross_instance",
            status="passed" if cross_instance_summary.passed else "failed",
            expected_contract={
                "minimum_required": draft.environment.cross_instance_set.minimum_required,
            },
            actual_data={
                "instance_count": cross_instance_summary.instance_count,
                "canonical_answer_count": cross_instance_summary.canonical_answer_count,
                "error_codes": list(cross_instance_summary.error_codes),
            },
            checks={"passed": cross_instance_summary.passed},
            diagnostics={"env_id": draft.environment.env_id},
        )
        if not cross_instance_summary.passed:
            return ProofEnvironmentRunSummary(
                db_id=draft.environment.db_id,
                env_id=draft.environment.env_id,
                fixture_sql_root=fixture_files.root_dir,
                quality_gate_status="reject_cross_instance",
                flow_id=flow_id,
                phase_monitor_log_path=phase_monitor_log_path,
                cross_instance_error_codes=cross_instance_summary.error_codes,
            )
        rollout_summary = await self.environment_orchestrator.run_draft(draft)
        phase_monitor.emit(
            phase="rollout",
            status="completed",
            expected_contract={
                "planned_solver_runs_upper_bound": self.config.calibration.full_replica_limit,
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
            diagnostics={"env_id": draft.environment.env_id},
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
                "accepted": quality_gate_summary.status
                is EnvironmentQualityGateStatus.ACCEPT,
            },
            diagnostics={"env_id": draft.environment.env_id},
        )
        if quality_gate_summary.status is not EnvironmentQualityGateStatus.ACCEPT:
            return ProofEnvironmentRunSummary(
                db_id=draft.environment.db_id,
                env_id=draft.environment.env_id,
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
        commit_result = self.registry.commit_draft(accepted_draft)
        phase_monitor.emit(
            phase="registry_commit",
            status=commit_result.status.value,
            expected_contract={},
            actual_data={
                "registry_env_id": commit_result.env_id,
                "status": commit_result.status.value,
            },
            checks={},
            diagnostics={"env_id": accepted_draft.environment.env_id},
        )
        bundle_root = output_root / "bundle"
        self.exporter.export_bundle(bundle_root, env_id=commit_result.env_id)
        phase_monitor.emit(
            phase="bundle_export",
            status="completed",
            expected_contract={"bundle_root": bundle_root},
            actual_data={"bundle_root": bundle_root, "env_id": commit_result.env_id},
            checks={"bundle_root_exists": bundle_root.exists()},
            diagnostics={},
        )
        return ProofEnvironmentRunSummary(
            db_id=accepted_draft.environment.db_id,
            env_id=accepted_draft.environment.env_id,
            fixture_sql_root=fixture_files.root_dir,
            quality_gate_status=quality_gate_summary.status.value,
            flow_id=flow_id,
            phase_monitor_log_path=phase_monitor_log_path,
            solver_pass_rate=quality_gate_summary.pass_rate,
            solver_ci_low=quality_gate_summary.ci_lower,
            solver_ci_high=quality_gate_summary.ci_upper,
            registry_status=commit_result.status,
            registry_env_id=commit_result.env_id,
            bundle_root=bundle_root,
        )

    async def close(self) -> None:
        await self.environment_orchestrator.close()


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


def build_proof_environment_draft(
    *,
    created_at: datetime | None = None,
) -> SynthesisEnvironmentDraft:
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
    solution_fingerprint = "sha256:" + sha256(canonical_answer_json.encode("utf-8")).hexdigest()

    task = TaskContract(
        question=(
            "봄 시즌 3일 출장 일정표를 만들어 주세요. 각 day는 하나의 city를 방문해야 하고 "
            "전체 itinerary에서 city는 중복되면 안 됩니다. day별 total_cost는 250 이하여야 "
            "하며, 연속된 day의 city는 인접한 지역이어야 합니다. 가능한 일정이 여러 개면 "
            "day 1의 city 이름, 그다음 day 2의 city 이름, 그다음 day 3의 city 이름의 "
            "사전순이 가장 앞서는 답을 고르세요."
        ),
        category=CategoryTaxonomy.ITINERARY,
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
        difficulty_vector=DifficultyVectorContract.model_validate(
            {
                "search_cost": 3.0,
                "solution_space": 3.0,
                "constraint_density": 4.0,
            }
        ),
        instance_parameters={
            "anchor_id": 1,
            "season": "spring",
            "budget_bucket": "mid",
        },
    )
    environment = EnvironmentContract(
        env_id=PROOF_ENV_ID,
        db_id=PROOF_DB_ID,
        domain="travel_planning",
        category=CategoryTaxonomy.ITINERARY,
        atomic_tool_set_ref=f"db://{PROOF_DB_ID}",
        difficulty_vector=task.difficulty_vector,
        created_at=created_at,
        generator_version=CURRENT_SYNTHESIS_GENERATOR_VERSION,
        tool_signature=_sha256_hex(_proof_atomic_tool_bundle().source),
        task_signature=_sha256_hex(task.model_dump_json()),
        verifier_signature=_sha256_hex(_proof_verifier_source()),
        status=EnvironmentStatus.DRAFT,
        quality_metrics=EnvironmentQualityMetrics(self_consistency_pass=True),
        rollout_constraints=RolloutConstraintsContract(
            max_turns=12,
            max_episode_duration_ms=90000,
            max_tool_rows=50,
        ),
        task=task,
        solution=SolutionContract(),
        verifier=VerifierContract(facts_schema=MaterializedFactsSchema()),
        shadow_verifier=ShadowVerifierContract(facts_schema=MaterializedFactsSchema()),
        instance_space=InstanceSpaceContract(
            anchor_query=AnchorQueryContract(
                sql=(
                    "SELECT anchor_id, season, budget_bucket "
                    "FROM proof_anchors ORDER BY anchor_id"
                ),
                outputs=["anchor_id", "season", "budget_bucket"],
            ),
            instance_count=1,
        ),
        cross_instance_set=CrossInstanceSet(
            minimum_required=1,
            instances=[
                InstanceContract(
                    instance_id="instance_0001",
                    anchor_values={
                        "anchor_id": 1,
                        "season": "spring",
                        "budget_bucket": "mid",
                    },
                    parameter_values={"day_count": 3},
                    expected_solution_fingerprint=solution_fingerprint,
                )
            ],
        ),
    )
    report = _passed_registration_report()
    rendered_prompt = build_rendered_user_prompt(task)
    return SynthesisEnvironmentDraft(
        created_at=created_at,
        db_id=PROOF_DB_ID,
        requested_category=CategoryTaxonomy.ITINERARY,
        schema_summary={"included_table_count": 5, "fixture": "proof_trip_fixture"},
        selected_category=CategoryTaxonomy.ITINERARY,
        environment=environment,
        atomic_tool_bundle=_proof_atomic_tool_bundle(),
        artifacts=GeneratedArtifactBundle(
            solution_source=_proof_solution_source(),
            verifier_source=_proof_verifier_source(),
            shadow_verifier_source=_proof_shadow_verifier_source(),
        ),
        registration_report=report,
        registration_diagnostics=build_registration_diagnostics(report),
        self_consistency_diagnostics=SynthesisSelfConsistencyDiagnostics(
            passed=True,
            answer=canonical_answer,
            verify_result=True,
            shadow_verify_result=True,
        ),
        instances=[
            MaterializedInstanceRecord(
                instance_id="instance_0001",
                rendered_user_prompt=rendered_prompt,
                params={"day_count": 3},
                anchor_values={
                    "anchor_id": 1,
                    "season": "spring",
                    "budget_bucket": "mid",
                },
            )
        ],
        canonical_answers=[
            MaterializedCanonicalAnswerRecord(
                instance_id="instance_0001",
                canonical_answer=canonical_answer,
                canonical_answer_json=canonical_answer_json,
                solution_fingerprint=solution_fingerprint,
            )
        ],
        provider_status={},
    )


def _proof_atomic_tool_bundle() -> AtomicToolBundle:
    tools = [
        AtomicToolDefinition(
            name="get_proof_anchor_by_id",
            family=AtomicToolFamily.T1_POINT_LOOKUP,
            description="Fetch one proof anchor row by anchor_id.",
            params_schema={
                "type": "object",
                "properties": {"anchor_id": {"type": "integer"}},
                "required": ["anchor_id"],
                "additionalProperties": False,
            },
            returns_schema={
                "type": ["object", "null"],
                "properties": {
                    "anchor_id": {"type": "integer"},
                    "season": {"type": "string"},
                    "budget_bucket": {"type": "string"},
                    "start_city_id": {"type": "integer"},
                },
                "required": ["anchor_id", "season", "budget_bucket", "start_city_id"],
            },
            sql=(
                "SELECT anchor_id, season, budget_bucket, start_city_id "
                "FROM proof_anchors WHERE anchor_id = $1 LIMIT 1"
            ),
            result_mode=AtomicToolResultMode.OBJECT_OR_NULL,
            semantic_key="proof_anchors:get_by_id",
        ),
        AtomicToolDefinition(
            name="list_proof_city_by_season_eq",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description="List proof cities visible in one season.",
            params_schema={
                "type": "object",
                "properties": {
                    "season": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["season", "limit"],
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
                "FROM proof_cities WHERE season = $1 ORDER BY city_id LIMIT $2"
            ),
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key="proof_cities:season_eq",
        ),
        AtomicToolDefinition(
            name="list_proof_city_link_by_city_id_eq",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description="List neighbor city ids for a proof city.",
            params_schema={
                "type": "object",
                "properties": {
                    "city_id": {"type": "integer"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["city_id", "limit"],
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
                "SELECT neighbor_city_id FROM proof_city_links "
                "WHERE city_id = $1 ORDER BY neighbor_city_id LIMIT $2"
            ),
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key="proof_city_links:city_id_eq",
        ),
        AtomicToolDefinition(
            name="traverse_proof_city_to_proof_lodging_via_city_id",
            family=AtomicToolFamily.T4_FK_TRAVERSAL,
            description="One-hop traversal from proof city to proof lodgings.",
            params_schema={
                "type": "object",
                "properties": {
                    "city_id": {"type": "integer"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["city_id", "limit"],
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
                "SELECT lodging_id, lodging_name, nightly_cost FROM proof_lodgings "
                "WHERE city_id = $1 ORDER BY lodging_id LIMIT $2"
            ),
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key="proof_cities->proof_lodgings:city_id",
        ),
        AtomicToolDefinition(
            name="traverse_proof_city_to_proof_activity_via_city_id",
            family=AtomicToolFamily.T4_FK_TRAVERSAL,
            description="One-hop traversal from proof city to proof activities.",
            params_schema={
                "type": "object",
                "properties": {
                    "city_id": {"type": "integer"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["city_id", "limit"],
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
                "SELECT activity_id, activity_name, ticket_cost FROM proof_activities "
                "WHERE city_id = $1 ORDER BY activity_id LIMIT $2"
            ),
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key="proof_cities->proof_activities:city_id",
        ),
    ]
    source = """
\"\"\"Atomic tools for the proof_trip_fixture database.\"\"\"
from __future__ import annotations


async def get_proof_anchor_by_id(conn, anchor_id):
    row = await conn.fetchrow(
        \"\"\"
        SELECT anchor_id, season, budget_bucket, start_city_id
        FROM proof_anchors
        WHERE anchor_id = $1
        LIMIT 1
        \"\"\",
        anchor_id,
    )
    return None if row is None else dict(row)


async def list_proof_city_by_season_eq(conn, season, limit):
    limit = min(limit, 20)
    rows = await conn.fetch(
        \"\"\"
        SELECT city_id, city_name, region_name
        FROM proof_cities
        WHERE season = $1
        ORDER BY city_id
        LIMIT $2
        \"\"\",
        season,
        limit,
    )
    return [dict(row) for row in rows]


async def list_proof_city_link_by_city_id_eq(conn, city_id, limit):
    limit = min(limit, 20)
    rows = await conn.fetch(
        \"\"\"
        SELECT neighbor_city_id
        FROM proof_city_links
        WHERE city_id = $1
        ORDER BY neighbor_city_id
        LIMIT $2
        \"\"\",
        city_id,
        limit,
    )
    return [dict(row) for row in rows]


async def traverse_proof_city_to_proof_lodging_via_city_id(conn, city_id, limit):
    limit = min(limit, 20)
    rows = await conn.fetch(
        \"\"\"
        SELECT lodging_id, lodging_name, nightly_cost
        FROM proof_lodgings
        WHERE city_id = $1
        ORDER BY lodging_id
        LIMIT $2
        \"\"\",
        city_id,
        limit,
    )
    return [dict(row) for row in rows]


async def traverse_proof_city_to_proof_activity_via_city_id(conn, city_id, limit):
    limit = min(limit, 20)
    rows = await conn.fetch(
        \"\"\"
        SELECT activity_id, activity_name, ticket_cost
        FROM proof_activities
        WHERE city_id = $1
        ORDER BY activity_id
        LIMIT $2
        \"\"\",
        city_id,
        limit,
    )
    return [dict(row) for row in rows]
""".strip() + "\n"
    return AtomicToolBundle(db_id=PROOF_DB_ID, tools=tools, source=source)


def _proof_solution_source() -> str:
    return """
def solve(tools):
    anchor = tools.get_proof_anchor_by_id({"anchor_id": 1})
    cities = tools.list_proof_city_by_season_eq({"season": anchor["season"], "limit": 20})
    plans = []
    for city in cities:
        lodgings = tools.traverse_proof_city_to_proof_lodging_via_city_id(
            {"city_id": city["city_id"], "limit": 20}
        )
        activities = tools.traverse_proof_city_to_proof_activity_via_city_id(
            {"city_id": city["city_id"], "limit": 20}
        )
        if not lodgings or not activities:
            continue
        plans.append(
            {
                "city_id": city["city_id"],
                "city": city["city_name"],
                "lodging": lodgings[0]["lodging_name"],
                "activity": activities[0]["activity_name"],
                "total_cost": lodgings[0]["nightly_cost"] + activities[0]["ticket_cost"],
                "neighbors": [
                    row["neighbor_city_id"]
                    for row in tools.list_proof_city_link_by_city_id_eq(
                        {"city_id": city["city_id"], "limit": 20}
                    )
                ],
            }
        )
    return plans
""".strip() + "\n"


def _proof_verifier_source() -> str:
    return """
async def fetch_facts(answer, tools):
    facts = []
    for item in answer:
        neighbors = tools.list_proof_city_link_by_city_id_eq(
            {"city_id": item["city_id"], "limit": 20}
        )
        facts.append({"city_id": item["city_id"], "neighbors": neighbors})
    return {"rows": facts}


def facts_match_answer_claims(answer, facts):
    return len(answer) == 3 and bool(facts["rows"])


def check_constraints(answer, facts):
    cities = [item["city"] for item in answer]
    if len(set(cities)) != len(cities):
        return False
    if any(item["total_cost"] > 250 for item in answer):
        return False
    return True


def verify(answer, tools):
    return isinstance(answer, list) and len(answer) == 3
""".strip() + "\n"


def _proof_shadow_verifier_source() -> str:
    return """
async def fetch_facts(answer, tools):
    rows = []
    for item in answer:
        rows.append(
            {
                "city_id": item["city_id"],
                "lodging_options": tools.traverse_proof_city_to_proof_lodging_via_city_id(
                    {"city_id": item["city_id"], "limit": 20}
                ),
            }
        )
    return {"rows": rows}


def facts_match_answer_claims(answer, facts):
    return len(answer) == 3 and len(facts["rows"]) == len(answer)


def check_constraints(answer, facts):
    return all(item["total_cost"] <= 250 for item in answer)


def verify(answer, tools):
    return isinstance(answer, list)
""".strip() + "\n"


def _passed_registration_report() -> RegistrationBundleReport:
    report = RegistrationBundleReport(
        status=RegistrationBundleStatus.PASSED,
        tool=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.TOOL,
            artifact_kind=ArtifactKind.TOOL_MODULE,
        ),
        tool_self_test=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.TOOL_SELF_TEST,
            artifact_kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
        ),
        solution=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.SOLUTION,
            artifact_kind=ArtifactKind.SOLUTION_MODULE,
        ),
        verifier=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.VERIFIER,
            artifact_kind=ArtifactKind.VERIFIER_MODULE,
        ),
        shadow_verifier=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.SHADOW_VERIFIER,
            artifact_kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
        ),
    )
    return report
def _sha256_hex(payload: str) -> str:
    return "sha256:" + sha256(payload.encode("utf-8")).hexdigest()
