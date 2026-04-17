from __future__ import annotations

from rl_task_foundry.infra.sdk_helpers import tool_choice_for_model


def test_tool_choice_is_required_for_gpt_nano() -> None:
    assert tool_choice_for_model("gpt-5.4-nano") == "required"


def test_tool_choice_is_required_for_claude() -> None:
    assert tool_choice_for_model("claude-sonnet-4-6") == "required"


def test_tool_choice_is_auto_for_qwen_thinking_model() -> None:
    assert tool_choice_for_model("qwen3.5-plus") == "auto"
    assert tool_choice_for_model("QWEN3.5-MAX") == "auto"
    assert tool_choice_for_model("Qwen3.6-Plus") == "auto"


def test_tool_choice_is_auto_for_deepseek_reasoning() -> None:
    assert tool_choice_for_model("deepseek-r1") == "auto"
    assert tool_choice_for_model("deepseek-v3-reasoning") == "auto"
