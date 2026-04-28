"""Tool-call budget prompt helper for the synthesis agent."""

from __future__ import annotations


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    first_submit_deadline = max(1, max_tool_calls // 3)
    return (
        "# Draft Submission Budget\n"
        f"Budget: {max_tool_calls} tool calls total, including data tools and "
        "`submit_draft`.\n"
        "\n"
        f"- Call `submit_draft` within your first {first_submit_deadline} "
        "tool calls.\n"
        "- Use at most 3 data-tool calls before first `submit_draft`.\n"
        "- If feedback says the draft needs more specificity, apply the "
        "Difficulty-Up Policy and resubmit.\n"
        "- Stop on overconstrained or terminal feedback.\n"
        "- No `submit_draft` before budget exhaustion ends the conversation.\n"
        "\n"
        "Why: early submission exposes schema/contract feedback before "
        "exploration consumes budget."
    )
