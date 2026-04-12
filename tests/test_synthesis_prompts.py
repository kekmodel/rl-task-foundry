from __future__ import annotations

import json

from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, DifficultyAxis
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_phase_input,
    build_synthesis_phase_instructions,
)
from rl_task_foundry.synthesis.runtime import SynthesisPhase, SynthesisStageRequest


def test_build_synthesis_phase_input_includes_atomic_tool_context() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        atomic_tool_set_ref="db://sakila",
        available_atomic_tools=[
            {
                "name": "get_customer_by_id",
                "description": "Lookup a customer by id.",
                "params_schema": {"type": "object"},
                "returns_schema": {"type": "object"},
            },
            {
                "name": "list_customer_ids",
                "description": "List customer ids.",
                "params_schema": {"type": "object"},
                "returns_schema": {"type": "array"},
            },
        ],
        domain_name="customer_support",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
    )

    payload = json.loads(build_synthesis_phase_input(request))

    assert payload["atomic_tool_set_ref"] == "db://sakila"
    assert payload["available_atomic_tool_names"] == [
        "get_customer_by_id",
        "list_customer_ids",
    ]
    assert payload["available_atomic_tools"][0]["name"] == "get_customer_by_id"
    assert "only tools" in payload["available_atomic_tools_role"]
    assert "tool.py" in payload["available_atomic_tools_role"]


def test_build_synthesis_phase_input_includes_difficulty_crank_context() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        atomic_tool_set_ref="db://sakila",
        domain_name="customer_support",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        strongest_difficulty_vector={
            DifficultyAxis.SLOT_COUNT: 3.0,
            DifficultyAxis.CONSTRAINT_COUNT: 4.0,
        },
        difficulty_crank_index=1,
        difficulty_crank_history=[DifficultyAxis.CONSTRAINT_COUNT],
    )

    payload = json.loads(build_synthesis_phase_input(request))

    assert payload["strongest_difficulty_vector"] == {
        "slot_count": 3.0,
        "constraint_count": 4.0,
    }
    assert payload["difficulty_crank_index"] == 1
    assert payload["difficulty_crank_history"] == ["constraint_count"]
    assert "Increase at most one difficulty axis" in payload["difficulty_crank_role"]


def test_schema_exploration_instructions_reference_atomic_tool_feasibility() -> None:
    instructions = build_synthesis_phase_instructions(SynthesisPhase.SCHEMA_EXPLORATION)

    assert "available atomic tools" in instructions
    assert "chaining those tools" in instructions
    assert "new tools can be created" in instructions


def test_artifact_generation_instructions_forbid_new_tool_generation() -> None:
    instructions = build_synthesis_phase_instructions(SynthesisPhase.ARTIFACT_GENERATION)

    assert "shared atomic tool set" in instructions
    assert "Do not generate `tool.py` or `tool_self_test.py`" in instructions
    assert "available_atomic_tools" in instructions
    assert "unique canonical JSON answer" in instructions
