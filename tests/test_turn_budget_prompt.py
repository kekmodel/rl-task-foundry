from __future__ import annotations

from rl_task_foundry.config.models import SynthesisRuntimeConfig
from rl_task_foundry.synthesis.prompts import build_synthesis_agent_instructions
from rl_task_foundry.synthesis.turn_budget import build_tool_call_budget_instruction


def test_budget_instruction_uses_tool_call_language_not_turns() -> None:
    text = build_tool_call_budget_instruction(max_tool_calls=18)

    assert "18 tool calls total" in text
    assert "submit_draft" in text
    assert "# Draft Submission Budget" in text
    # Never say "turns" in the budget text — keep vocabulary concrete.
    assert "turn" not in text.lower().split("tool calls")[0]
    assert "data tools" in text
    assert "After feedback, max 3 data tools" in text
    assert "binding feedback uses none" in text
    assert "only label `query`" in text
    assert "ToolBudgetFeedback" in text
    assert "submit next" in text
    for leaked in ("composer tools", "too_easy", "too_hard", "trial"):
        assert leaked not in text


def test_budget_instruction_derives_first_submit_deadline_as_third_of_budget() -> None:
    text = build_tool_call_budget_instruction(max_tool_calls=18)

    # 18 // 3 == 6
    assert "first 6 tool calls" in text


def test_budget_instruction_floor_is_at_least_one_even_for_tiny_budgets() -> None:
    text = build_tool_call_budget_instruction(max_tool_calls=2)

    assert "first 1 tool calls" in text


def test_synthesis_instructions_embed_the_budget_block() -> None:
    runtime_config = SynthesisRuntimeConfig(max_turns=21)
    text = build_synthesis_agent_instructions(runtime_config)

    assert "21 tool calls" in text
    assert "submit_draft" in text
    assert "# Draft Submission Budget" in text


def test_synthesis_instructions_contain_hard_prohibitions() -> None:
    runtime_config = SynthesisRuntimeConfig(max_turns=20)
    text = build_synthesis_agent_instructions(runtime_config)

    for prohibition in (
        "never raw wording",
        "Do not invent ids",
        "Do not reformat",
    ):
        assert prohibition in text


def test_synthesis_instructions_defer_submit_draft_shape_to_tool_schema() -> None:
    runtime_config = SynthesisRuntimeConfig(max_turns=20)
    text = build_synthesis_agent_instructions(runtime_config)

    assert "# submit_draft" not in text
    assert "topic = " not in text
    assert "entity = " not in text
    assert "follow that tool's schema exactly" in text
