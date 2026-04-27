"""Tool-call budget prompt helper for the synthesis agent."""

from __future__ import annotations


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    first_submit_deadline = max(1, max_tool_calls // 3)
    return (
        "# Draft Submission Budget\n"
        f"You have a budget of {max_tool_calls} tool calls total. The budget "
        "includes data tools and `submit_draft`.\n"
        "\n"
        f"- Call `submit_draft` at least once within your first "
        f"{first_submit_deadline} tool calls.\n"
        "- Spend at most 3 data-tool calls before the first `submit_draft`.\n"
        "- When feedback says the draft needs more specificity, apply the "
        "Difficulty-Up Policy and resubmit.\n"
        "- On overconstrained or terminal feedback, stop.\n"
        f"- Exhausting the {max_tool_calls}-call budget without `submit_draft` "
        "ends the conversation.\n"
        "\n"
        "Why: early submission exposes schema and contract feedback before "
        "exploration consumes the budget."
    )
