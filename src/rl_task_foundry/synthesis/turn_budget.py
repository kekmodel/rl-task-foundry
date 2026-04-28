"""Tool-call budget prompt helper for the synthesis agent."""

from __future__ import annotations


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    first_submit_deadline = max(1, max_tool_calls // 3)
    return (
        "# Draft Submission Budget\n"
        f"Budget: {max_tool_calls} tool calls total, incl. data tools and "
        "`submit_draft`.\n"
        "\n"
        f"- Call `submit_draft` within your first {first_submit_deadline} "
        "tool calls.\n"
        "- Max 3 data tools before first `submit_draft`.\n"
        "- Specificity feedback: reminder for Difficulty-Up Policy.\n"
        "- If `submit_draft` says the conversation is terminated, stop.\n"
        "- No `submit_draft` before budget exhaustion ends conversation."
    )
