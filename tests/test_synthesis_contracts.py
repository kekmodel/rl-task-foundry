from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rl_task_foundry.synthesis.contracts import (
    AnchorQueryContract,
    CategoryTaxonomy,
    ConstraintKind,
    ConstraintSummaryItem,
    CrossInstanceSet,
    DifficultyAxis,
    DIFFICULTY_AXIS_SPECS,
    DIFFICULTY_CRANK_ORDER,
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnumParameterSpace,
    FloatRangeParameterSpace,
    InstanceContract,
    InstanceSamplingContract,
    InstanceSamplingStrategy,
    InstanceSpaceContract,
    IntRangeParameterSpace,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    SolutionContract,
    TaskContract,
    ToolContract,
    ToolEmptyResultBehavior,
    ToolParameterContract,
    ToolParameterType,
    ToolSelfTestCheck,
    ToolSelfTestContract,
    ToolTimeoutBehavior,
    build_difficulty_vector,
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


def test_instance_space_supports_contract_shapes() -> None:
    instance_space = InstanceSpaceContract(
        anchor_query=AnchorQueryContract(
            sql="SELECT itinerary_anchor_id, season FROM proof_anchors ORDER BY itinerary_anchor_id",
            outputs=["itinerary_anchor_id", "season"],
        ),
        parameters={
            "budget_bucket": EnumParameterSpace(values=["low", "mid", "high"]),
            "day_count": IntRangeParameterSpace(min=2, max=4),
        },
        sampling=InstanceSamplingContract(
            strategy=InstanceSamplingStrategy.DETERMINISTIC_HASH,
            seed=17,
        ),
        instance_count=3,
    )

    assert instance_space.instance_count == 3
    assert instance_space.parameters["budget_bucket"].kind == "enum"
    assert instance_space.parameters["day_count"].kind == "int_range"


def test_environment_contract_round_trips_without_verifier_fields() -> None:
    output_schema = _build_output_schema()
    difficulty_vector = build_difficulty_vector(
        search_cost=2.0,
        solution_space=3.0,
        constraint_density=4.0,
    )
    task = TaskContract(
        question="3일 일정표를 만들어 주세요.",
        category=CategoryTaxonomy.ITINERARY,
        output_schema=output_schema,
        constraint_summary=[
            ConstraintSummaryItem(
                key="no_repeat_city",
                kind=ConstraintKind.UNIQUENESS,
                summary="도시는 전체 일정에서 중복되면 안 된다.",
            )
        ],
        difficulty_vector=difficulty_vector,
        instance_parameters={"budget_bucket": "mid"},
    )
    contract = EnvironmentContract(
        env_id="env_trip_fixture_v1",
        db_id="proof_trip_fixture",
        domain="travel",
        category=CategoryTaxonomy.ITINERARY,
        atomic_tool_set_ref="db://proof_trip_fixture",
        difficulty_vector=difficulty_vector,
        created_at=datetime(2026, 4, 11, 13, 40, 0, tzinfo=timezone.utc),
        generator_version="rewrite-v1",
        tool_signature="toolhash",
        task_signature="taskhash",
        quality_metrics=EnvironmentQualityMetrics(),
        rollout_constraints=RolloutConstraintsContract(
            max_turns=16,
            max_episode_duration_ms=80000,
            max_tool_rows=100,
        ),
        task=task,
        solution=SolutionContract(),
        instance_space=InstanceSpaceContract(
            anchor_query=AnchorQueryContract(
                sql="SELECT anchor_id FROM proof_anchors ORDER BY anchor_id",
                outputs=["anchor_id"],
            )
        ),
        cross_instance_set=CrossInstanceSet(
            instances=[
                InstanceContract(instance_id="instance_1"),
                InstanceContract(instance_id="instance_2"),
            ]
        ),
    )

    round_tripped = EnvironmentContract.model_validate_json(contract.model_dump_json())

    assert round_tripped.task.category is CategoryTaxonomy.ITINERARY
    assert round_tripped.atomic_tool_set_ref == "db://proof_trip_fixture"
    assert round_tripped.rollout_constraints.max_tool_rows == 100
    assert round_tripped.quality_metrics.solver_pass_rate is None


def test_environment_contract_rejects_task_category_mismatch() -> None:
    output_schema = _build_output_schema()
    task = TaskContract(
        question="일정표를 만들어 주세요.",
        category=CategoryTaxonomy.ASSIGNMENT,
        output_schema=output_schema,
        difficulty_vector=build_difficulty_vector(solution_space=3.0),
    )

    with pytest.raises(ValidationError):
        EnvironmentContract(
            env_id="env_bad",
            db_id="proof_trip_fixture",
            domain="travel",
            category=CategoryTaxonomy.ITINERARY,
            atomic_tool_set_ref="db://proof_trip_fixture",
            difficulty_vector=build_difficulty_vector(solution_space=3.0),
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
            solution=SolutionContract(),
            instance_space=InstanceSpaceContract(
                anchor_query=AnchorQueryContract(
                    sql="SELECT anchor_id FROM proof_anchors ORDER BY anchor_id",
                    outputs=["anchor_id"],
                )
            ),
        )


def test_tool_contract_defaults_error_behavior() -> None:
    tool = ToolContract(
        name="get_hotels_by_city",
        return_schema=OutputFieldContract(
            name="hotels",
            type=OutputFieldType.LIST,
            items=OutputFieldContract(name="hotel", type=OutputFieldType.STRING),
        ),
    )

    assert tool.empty_result_behavior is ToolEmptyResultBehavior.RETURN_EMPTY
    assert tool.timeout_behavior is ToolTimeoutBehavior.RAISE_TIMEOUT


def test_tool_self_test_contract_defaults_required_checks() -> None:
    contract = ToolSelfTestContract()

    assert contract.entrypoint == "run_self_test"
    assert contract.required_checks == [
        ToolSelfTestCheck.HAPPY_PATH,
        ToolSelfTestCheck.EMPTY_RESULT_BEHAVIOR,
        ToolSelfTestCheck.TIMEOUT_BEHAVIOR,
        ToolSelfTestCheck.DETERMINISTIC_ORDERING,
    ]


def test_difficulty_axis_specs_cover_every_axis_in_crank_order() -> None:
    assert set(DIFFICULTY_AXIS_SPECS) == set(DifficultyAxis)
    assert tuple(DIFFICULTY_AXIS_SPECS) == DIFFICULTY_CRANK_ORDER


def test_float_range_parameter_space_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        FloatRangeParameterSpace(min=10.0, max=5.0)


def test_tool_parameter_contract_requires_enum_values_for_enum_types() -> None:
    with pytest.raises(ValidationError):
        ToolParameterContract(name="bucket", type=ToolParameterType.ENUM)


def test_cross_instance_set_rejects_duplicate_solution_fingerprints_when_required() -> None:
    with pytest.raises(ValidationError):
        CrossInstanceSet(
            instances=[
                InstanceContract(
                    instance_id="instance_1",
                    expected_solution_fingerprint="sha256:abc",
                ),
                InstanceContract(
                    instance_id="instance_2",
                    expected_solution_fingerprint="sha256:abc",
                ),
            ]
        )
