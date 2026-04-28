"""Tool-call budget prompt helper for the synthesis agent."""

from __future__ import annotations

FIRST_SUBMIT_MAX_DATA_TOOLS = 3
FEEDBACK_REPAIR_MAX_DATA_TOOLS = 2


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    first_submit_deadline = max(1, max_tool_calls // 3)
    return (
        "# Draft Submission Budget\n"
        f"Budget: {max_tool_calls} tool calls total incl. data tools and "
        "`submit_draft`.\n"
        "\n"
        f"- Call `submit_draft` within first {first_submit_deadline} "
        "tool calls.\n"
        f"- Max {FIRST_SUBMIT_MAX_DATA_TOOLS} data tools before first "
        "`submit_draft`.\n"
        f"- After feedback, use at most {FEEDBACK_REPAIR_MAX_DATA_TOOLS} data "
        "tools before next `submit_draft`; at limit, only label `query` if "
        "rows missing; submit when rows return.\n"
        "- Specificity feedback reminds Difficulty-Up Policy.\n"
        "- If `submit_draft` says the conversation is terminated, stop.\n"
        "- Plain text is invalid; call a tool every turn."
    )
