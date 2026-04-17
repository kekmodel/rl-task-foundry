from __future__ import annotations

from rl_task_foundry.config.models import SynthesisRuntimeConfig
from rl_task_foundry.synthesis.prompts import build_synthesis_agent_instructions
from rl_task_foundry.synthesis.turn_budget import build_tool_call_budget_instruction


def test_budget_instruction_uses_tool_call_language_not_turns() -> None:
    text = build_tool_call_budget_instruction(max_tool_calls=18)

    assert "18 tool calls" in text
    assert "submit_draft" in text
    assert "# Commit Rule" in text
    # Never say "turns" in the budget text — keep vocabulary concrete.
    assert "turn" not in text.lower().split("tool calls")[0]


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
    assert "# Commit Rule" in text


def test_synthesis_instructions_contain_never_block_with_hard_prohibitions() -> None:
    runtime_config = SynthesisRuntimeConfig(max_turns=20)
    text = build_synthesis_agent_instructions(runtime_config)

    assert "# Never" in text
    for prohibition in (
        "Never write SQL",
        "Never weaken",
        "Never concatenate",
    ):
        assert prohibition in text


def test_synthesis_instructions_embed_submit_draft_format_block() -> None:
    runtime_config = SynthesisRuntimeConfig(max_turns=20)
    text = build_synthesis_agent_instructions(runtime_config)

    assert "# submit_draft" in text
    assert "topic = " in text
    assert "entity = " in text
    assert "label = " in text
    assert "question = " in text
