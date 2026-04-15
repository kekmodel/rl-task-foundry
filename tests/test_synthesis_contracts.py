from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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


def _build_output_schema() -> OutputSchemaContract:
    return OutputSchemaContract(
        root=OutputFieldContract(
            name="days",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("date", "city", "hotel"),
            items=OutputFieldContract(
                name="day",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="date", type=OutputFieldType.DATE),
                    OutputFieldContract(name="city", type=OutputFieldType.STRING),
                    OutputFieldContract(name="hotel", type=OutputFieldType.STRING),
                ],
            ),
        ),
        primary_output_format="json_array",
    )


def test_output_schema_supports_nested_list_objects() -> None:
    schema = _build_output_schema()

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.items is not None
    assert schema.root.items.type is OutputFieldType.OBJECT


def test_output_schema_rejects_invalid_scalar_children() -> None:
    with pytest.raises(ValidationError):
        OutputFieldContract(
            name="city",
            type=OutputFieldType.STRING,
            fields=[OutputFieldContract(name="nested", type=OutputFieldType.STRING)],
        )


def test_task_bundle_contract_round_trips_without_generated_artifacts() -> None:
    output_schema = _build_output_schema()
    task = TaskContract(
        question="3일 일정표를 만들어 주세요.",
        topic="itinerary",
        output_schema=output_schema,
        constraint_summary=[
            ConstraintSummaryItem(
                key="no_repeat_city",
                kind=ConstraintKind.UNIQUENESS,
                summary="도시는 전체 일정에서 중복되면 안 된다.",
            )
        ],
        instance_parameters={"budget_bucket": "mid"},
    )
    contract = TaskBundleContract(
        task_id="task_trip_fixture_v1",
        db_id="proof_trip_fixture",
        domain="travel",
        topic="itinerary",
        atomic_tool_set_ref="db://proof_trip_fixture",
        created_at=datetime(2026, 4, 11, 13, 40, 0, tzinfo=timezone.utc),
        generator_version="rewrite-v1",
        tool_signature="toolhash",
        task_signature="taskhash",
        status=TaskBundleStatus.DRAFT,
        quality_metrics=TaskQualityMetrics(),
        rollout_constraints=RolloutConstraintsContract(
            max_turns=16,
            max_episode_duration_ms=80000,
            max_tool_rows=100,
        ),
        task=task,
    )

    round_tripped = TaskBundleContract.model_validate_json(contract.model_dump_json())

    assert round_tripped.task.topic == "itinerary"
    assert round_tripped.atomic_tool_set_ref == "db://proof_trip_fixture"
    assert round_tripped.rollout_constraints.max_tool_rows == 100
    assert round_tripped.quality_metrics.solver_pass_rate is None


def test_task_bundle_contract_rejects_task_topic_mismatch() -> None:
    output_schema = _build_output_schema()
    task = TaskContract(
        question="일정표를 만들어 주세요.",
        topic="assignment",
        output_schema=output_schema,
    )

    with pytest.raises(ValidationError):
        TaskBundleContract(
            task_id="task_bad",
            db_id="proof_trip_fixture",
            domain="travel",
            topic="itinerary",
            atomic_tool_set_ref="db://proof_trip_fixture",
            created_at=datetime(2026, 4, 11, 13, 40, 0, tzinfo=timezone.utc),
            generator_version="rewrite-v1",
            tool_signature="toolhash",
            task_signature="taskhash",
            rollout_constraints=RolloutConstraintsContract(
                max_turns=16,
                max_episode_duration_ms=80000,
                max_tool_rows=100,
            ),
            task=task,
        )


