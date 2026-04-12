from __future__ import annotations

from rl_task_foundry.synthesis.contracts import build_difficulty_vector
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_phase_input,
    build_synthesis_phase_instructions,
)
from rl_task_foundry.synthesis.runtime import (
    LabelConstructionOutput,
    SchemaExplorationOutput,
    SynthesisPhase,
    SynthesisQualityGateFeedback,
    SynthesisStageRequest,
)


def test_schema_exploration_prompt_keeps_tool_json_out_of_user_message() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.SCHEMA_EXPLORATION,
        db_id="sakila",
        atomic_tool_set_ref="db://sakila",
        available_atomic_tools=[
            {
                "name": "get_customer_by_id",
                "description": "Get customer for given primary key.",
                "params_schema": {"type": "object"},
                "returns_schema": {"type": "object"},
            }
        ],
        domain_name="customer_support",
        task_language="ko",
        scenario_description="help requests",
        requested_topic="assignment",
        schema_summary={
            "table_count": 2,
            "edge_count": 1,
            "tables": [
                {
                    "qualified_name": "public.customer",
                    "primary_key": ["customer_id"],
                    "column_names": ["customer_id", "name"],
                }
            ],
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "# Domain" in prompt
    assert "# Schema Orientation" in prompt
    assert "# Required Output Contract" in prompt
    assert "public.customer" in prompt
    assert "available_atomic_tools" not in prompt
    assert "atomic_tool_set_ref" not in prompt
    assert "difficulty_crank_index" not in prompt


def test_task_synthesis_prompt_includes_language_policy_and_one_shot() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.TASK_SYNTHESIS,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        scenario_description="help requests",
        requested_topic="assignment",
        previous_outputs={
            SynthesisPhase.SCHEMA_EXPLORATION: SchemaExplorationOutput(
                domain_hypothesis="customer support",
                candidate_topics=["assignment"],
                sample_observations=["customer 148 belongs to store 1"],
                memory_summary="schema done",
            ),
            SynthesisPhase.LABEL_CONSTRUCTION: LabelConstructionOutput(
                canonical_answer_json='{"store_id": 1}',
                anchor_entity={"customer_id": 148},
                difficulty_vector=build_difficulty_vector(search_cost=2.0),
                instance_parameters={},
                label_summary="grounded answer",
                memory_summary="label complete",
            ),
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "# User-Facing Language" in prompt
    assert "Korean" in prompt
    assert "All schema field names, code identifiers, and tool names must remain in English." in prompt
    assert "# One-Shot Example" in prompt
    assert '"question"' in prompt
    assert '"memory_summary"' in prompt
    assert '"label"' not in prompt
    assert "solution.py" not in prompt
    assert "verifier.py" not in prompt


def test_task_synthesis_prompt_includes_quality_feedback_without_debug_dump() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.TASK_SYNTHESIS,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        scenario_description="help requests",
        requested_topic="assignment",
        crank_hint="Increase search_cost by adding a new table join.",
        latest_quality_gate_feedback=SynthesisQualityGateFeedback(
            status="reject_too_easy",
            pass_rate=1.0,
            ci_lower=0.6,
            ci_upper=1.0,
            matched_solver_runs=4,
            total_solver_runs=4,
            previous_question="가장 쉬운 질문",
        ),
    )

    prompt = build_synthesis_phase_input(request)

    assert "# Difficulty Guidance" in prompt
    assert "Increase search_cost" in prompt
    assert "latest_quality_gate_feedback" not in prompt


def test_prompt_instructions_cover_only_four_phases() -> None:
    assert "data exploration phase" in build_synthesis_phase_instructions(
        SynthesisPhase.SCHEMA_EXPLORATION
    )
    assert "grounded topic string" in build_synthesis_phase_instructions(
        SynthesisPhase.CATEGORY_INFERENCE
    )
    assert "Construct the latent label first" in build_synthesis_phase_instructions(
        SynthesisPhase.LABEL_CONSTRUCTION
    )
    assert "Reverse-design the natural user request" in build_synthesis_phase_instructions(
        SynthesisPhase.TASK_SYNTHESIS
    )
