from __future__ import annotations

from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
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
    TaskSynthesisOutput,
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
        requested_category=CategoryTaxonomy.ASSIGNMENT,
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
    assert "# Exploration Goal" in prompt
    assert "# Required Output Contract" in prompt
    assert "public.customer" in prompt
    assert "available_atomic_tools" not in prompt
    assert "atomic_tool_set_ref" not in prompt
    assert "difficulty_crank_index" not in prompt


def test_task_synthesis_prompt_includes_language_policy_for_solution_only() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.TASK_SYNTHESIS,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        previous_outputs={
            SynthesisPhase.LABEL_CONSTRUCTION: LabelConstructionOutput(
                canonical_answer_json='{"store_id": 1}',
                output_schema={
                    "version": "v2",
                    "root": {
                        "name": "answer",
                        "type": "object",
                        "fields": [{"name": "store_id", "type": "int"}],
                    },
                    "primary_output_format": "json_object",
                },
                difficulty_vector=build_difficulty_vector(search_cost=2.0),
                instance_parameters={},
                label_summary="grounded answer",
                memory_summary="label complete",
            )
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "# User-Facing Language" in prompt
    assert "Generate the user-facing question" in prompt
    assert "Korean" in prompt
    assert "All code (solution.py), field names, and tool calls must remain in English." in prompt
    assert "verifier.py" not in prompt
    assert "shadow_verifier.py" not in prompt


def test_artifact_generation_prompt_uses_solution_only_contract() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        atomic_tool_set_ref="db://sakila",
        available_atomic_tools=[
            {
                "name": "get_customer_by_id",
                "description": "Get customer for given primary key.",
                "params_schema": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "integer"}},
                },
                "returns_schema": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"customer_id": {"type": "integer"}},
                        },
                        {"type": "null"},
                    ]
                },
            }
        ],
        domain_name="customer_support",
        task_language="ko",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        previous_outputs={
            SynthesisPhase.SCHEMA_EXPLORATION: SchemaExplorationOutput(
                domain_hypothesis="customer support",
                candidate_categories=[CategoryTaxonomy.ASSIGNMENT],
                sample_observations=["get_customer_by_id(148) returned customer_id 148"],
                memory_summary="schema done",
            ),
            SynthesisPhase.LABEL_CONSTRUCTION: LabelConstructionOutput(
                canonical_answer_json='{"store_id": 1}',
                output_schema={
                    "version": "v2",
                    "root": {
                        "name": "answer",
                        "type": "object",
                        "fields": [{"name": "store_id", "type": "int"}],
                    },
                    "primary_output_format": "json_object",
                },
                difficulty_vector=build_difficulty_vector(search_cost=2.0),
                instance_parameters={},
                label_summary="grounded answer",
                memory_summary="label complete",
            ),
            SynthesisPhase.TASK_SYNTHESIS: TaskSynthesisOutput(
                question="질문",
                constraint_summary=[],
                instance_space={
                    "anchor_query": {
                        "sql": "SELECT customer_id FROM customer",
                        "outputs": ["customer_id"],
                    },
                    "parameters": {},
                    "sampling": {"strategy": "deterministic_hash", "seed": 0},
                    "instance_count": 1,
                },
                memory_summary="task complete",
            ),
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "# Relevant Atomic Tools" in prompt
    assert "get_customer_by_id(customer_id)" in prompt
    assert "Generate solution.py" in prompt
    assert "verifier.py" not in prompt
    assert "shadow_verifier.py" not in prompt
    assert "\"solution_source\": \"python source string\"" in prompt


def test_artifact_generation_prompt_includes_quality_feedback_without_debug_dump() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
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
    assert "reject_too_easy" in prompt
    assert "Increase search_cost" in prompt
    assert "latest_quality_gate_feedback" not in prompt


def test_artifact_generation_instructions_are_solution_only() -> None:
    instructions = build_synthesis_phase_instructions(SynthesisPhase.ARTIFACT_GENERATION)

    assert "Generate only `solution.py`." in instructions
    assert "verifier" not in instructions
