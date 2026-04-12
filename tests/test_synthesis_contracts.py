from __future__ import annotations

from datetime import datetime

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
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnumParameterSpace,
    FactCardinality,
    FactValueType,
    FactSpec,
    FloatRangeParameterSpace,
    InstanceContract,
    InstanceSamplingContract,
    InstanceSamplingStrategy,
    InstanceSpaceContract,
    IntRangeParameterSpace,
    MaterializedFactsSchema,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    ShadowPromptStrategy,
    ShadowVerifierContract,
    SolutionContract,
    TaskContract,
    ToolContract,
    ToolEmptyResultBehavior,
    ToolParameterContract,
    ToolParameterType,
    ToolSelfTestCheck,
    ToolSelfTestContract,
    ToolTimeoutBehavior,
    VerifierContract,
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


def _build_facts_schema() -> MaterializedFactsSchema:
    return MaterializedFactsSchema(
        facts=[
            FactSpec(
                key="hotel_price",
                entity_ref="day_2.hotel",
                attribute="price",
                value_type=FactValueType.FLOAT,
                cardinality=FactCardinality.ONE,
            ),
            FactSpec(
                key="restaurant_ratings",
                entity_ref="day_2.restaurants",
                attribute="rating",
                value_type=FactValueType.FLOAT,
                nullable=False,
                cardinality=FactCardinality.MANY,
            ),
        ]
    )


def test_output_schema_supports_nested_list_objects() -> None:
    schema = _build_output_schema()

    dumped = schema.model_dump(mode="json")

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.items is not None
    assert schema.root.items.type is OutputFieldType.OBJECT
    assert dumped["primary_output_format"] == "json_array"


def test_output_schema_rejects_invalid_scalar_children() -> None:
    with pytest.raises(ValidationError):
        OutputFieldContract(
            name="city",
            type=OutputFieldType.STRING,
            fields=[OutputFieldContract(name="nested", type=OutputFieldType.STRING)],
        )


def test_output_schema_requires_enum_values_for_enum_fields() -> None:
    with pytest.raises(ValidationError):
        OutputFieldContract(name="tier", type=OutputFieldType.ENUM)


def test_tool_parameter_contract_requires_enum_values_for_enum_types() -> None:
    with pytest.raises(ValidationError):
        ToolParameterContract(name="bucket", type=ToolParameterType.ENUM)


def test_tool_parameter_contract_rejects_enum_values_for_non_enum_types() -> None:
    with pytest.raises(ValidationError):
        ToolParameterContract(
            name="city",
            type=ToolParameterType.STRING,
            enum_values=["seoul"],
        )


def test_fact_spec_rejects_unknown_value_type() -> None:
    with pytest.raises(ValidationError):
        FactSpec(
            key="bad",
            entity_ref="slot.hotel",
            attribute="price",
            value_type="weird_thing",
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


def test_anchor_query_contract_requires_unique_outputs() -> None:
    with pytest.raises(ValidationError):
        AnchorQueryContract(
            sql="SELECT anchor_id FROM proof_anchors ORDER BY anchor_id",
            outputs=["anchor_id", "anchor_id"],
        )


def test_float_range_parameter_space_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        FloatRangeParameterSpace(min=10.0, max=5.0)


def test_environment_contract_round_trips_with_core_artifacts() -> None:
    output_schema = _build_output_schema()
    difficulty_vector = {
        DifficultyAxis.SLOT_COUNT: 3.0,
        DifficultyAxis.CONSTRAINT_COUNT: 7.0,
        DifficultyAxis.CONDITIONAL_DEPTH: 1.0,
    }
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
    tool = ToolContract(
        name="get_hotels_by_city",
        description="Return hotels available in the requested city.",
        parameters=[
            ToolParameterContract(name="city", type=ToolParameterType.STRING),
        ],
        return_schema=OutputFieldContract(
            name="hotels",
            type=OutputFieldType.LIST,
            ordered=True,
            items=OutputFieldContract(
                name="hotel",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="name", type=OutputFieldType.STRING),
                    OutputFieldContract(name="price", type=OutputFieldType.FLOAT),
                ],
            ),
        ),
    )
    verifier = VerifierContract(facts_schema=_build_facts_schema())
    shadow_verifier = ShadowVerifierContract(
        facts_schema=_build_facts_schema(),
        prompt_strategy=ShadowPromptStrategy.BOTTOM_UP,
    )
    contract = EnvironmentContract(
        env_id="env_trip_fixture_v1",
        db_id="proof_trip_fixture",
        domain="travel",
        category=CategoryTaxonomy.ITINERARY,
        difficulty_vector=difficulty_vector,
        created_at=datetime(2026, 4, 11, 13, 40, 0),
        generator_version="rewrite-v1",
        tool_signature="toolhash",
        task_signature="taskhash",
        verifier_signature="verifierhash",
        quality_metrics=EnvironmentQualityMetrics(self_consistency_pass=True),
        tools=[tool],
        task=task,
        solution=SolutionContract(),
        verifier=verifier,
        shadow_verifier=shadow_verifier,
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
                InstanceContract(instance_id="instance_3"),
            ]
        ),
    )

    round_tripped = EnvironmentContract.model_validate_json(contract.model_dump_json())

    assert round_tripped.task.category is CategoryTaxonomy.ITINERARY
    assert round_tripped.shadow_verifier.official_judgment is False
    assert round_tripped.verifier.fetch_facts_function == "fetch_facts"
    assert round_tripped.tool_self_test.entrypoint == "run_self_test"
    assert round_tripped.quality_metrics.shadow_disagreement_rate is None


def test_task_contract_rejects_fractional_integral_difficulty_axis() -> None:
    with pytest.raises(ValidationError):
        TaskContract(
            question="일정표를 만들어 주세요.",
            category=CategoryTaxonomy.ITINERARY,
            output_schema=_build_output_schema(),
            difficulty_vector={DifficultyAxis.SLOT_COUNT: 2.5},
        )


def test_difficulty_axis_specs_cover_every_axis() -> None:
    assert set(DIFFICULTY_AXIS_SPECS) == set(DifficultyAxis)


def test_environment_contract_rejects_task_category_mismatch() -> None:
    output_schema = _build_output_schema()
    task = TaskContract(
        question="일정표를 만들어 주세요.",
        category=CategoryTaxonomy.ASSIGNMENT,
        output_schema=output_schema,
        difficulty_vector={DifficultyAxis.SLOT_COUNT: 3.0},
    )

    with pytest.raises(ValidationError):
        EnvironmentContract(
            env_id="env_bad",
            db_id="proof_trip_fixture",
            domain="travel",
            category=CategoryTaxonomy.ITINERARY,
            difficulty_vector={DifficultyAxis.SLOT_COUNT: 3.0},
            created_at=datetime(2026, 4, 11, 13, 40, 0),
            generator_version="rewrite-v1",
            tool_signature="toolhash",
            task_signature="taskhash",
            verifier_signature="verifierhash",
            task=task,
            solution=SolutionContract(),
            verifier=VerifierContract(facts_schema=_build_facts_schema()),
            shadow_verifier=ShadowVerifierContract(facts_schema=_build_facts_schema()),
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


def test_cross_instance_set_requires_unique_instance_ids() -> None:
    with pytest.raises(ValidationError):
        CrossInstanceSet(
            instances=[
                InstanceContract(instance_id="instance_1"),
                InstanceContract(instance_id="instance_1"),
            ]
        )


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


def test_environment_quality_metrics_allow_unmeasured_state() -> None:
    metrics = EnvironmentQualityMetrics()

    assert metrics.shadow_disagreement_rate is None
    assert metrics.solver_pass_rate is None
    assert metrics.solver_ci_low is None
    assert metrics.solver_ci_high is None


def test_environment_quality_metrics_require_both_ci_bounds_together() -> None:
    with pytest.raises(ValidationError):
        EnvironmentQualityMetrics(solver_ci_low=0.2)


def test_instance_contract_rejects_non_serializable_anchor_values() -> None:
    class _Sentinel:
        pass

    with pytest.raises(ValidationError):
        InstanceContract(
            instance_id="instance_1",
            anchor_values={"customer_id": _Sentinel()},
        )
