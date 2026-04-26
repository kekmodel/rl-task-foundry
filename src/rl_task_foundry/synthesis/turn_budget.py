"""Tool-call budget prompt helper for the synthesis agent."""

from __future__ import annotations


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    first_submit_deadline = max(1, max_tool_calls // 3)
    return (
        "# Commit Rule\n"
        f"You have a budget of {max_tool_calls} tool calls. Every tool call "
        "counts the same: data tools (schema_map/profile/"
        "sample/neighborhood/query) and "
        "submit_draft alike.\n"
        "\n"
        f"- Within your first {first_submit_deadline} tool calls, "
        "submit_draft must have been called at least once.\n"
        "- Spend at most 3 data-tool calls before your first submit.\n"
        "- A specificity rejection is feedback, not failure. Keep "
        "the previous draft and resubmit with ONE structural change. "
        "An overconstrained rejection ends the conversation.\n"
        f"- Exhausting the {max_tool_calls}-call budget without a submit "
        "ends the conversation.\n"
        "\n"
        "Submit early, iterate on rejection. Exploration is cheap after the "
        "first submit; exploration without a submit is waste."
    )
