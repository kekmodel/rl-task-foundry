"""Tool-call budget prompt helper for the synthesis agent."""

from __future__ import annotations

FIRST_SUBMIT_MAX_DATA_TOOLS = 3
FEEDBACK_REPAIR_MAX_DATA_TOOLS = 3


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    first_submit_deadline = max(1, max_tool_calls // 3)
    return (
        "# Draft Submission Budget\n"
        f"{max_tool_calls} tool calls total: data tools + submit_draft.\n"
        f"- submit_draft by first {first_submit_deadline} "
        "tool calls; "
        f"max {FIRST_SUBMIT_MAX_DATA_TOOLS} data tools.\n"
        f"- After feedback, max {FEEDBACK_REPAIR_MAX_DATA_TOOLS} data tools; "
        "binding feedback uses none.\n"
        "- ToolBudgetFeedback boundary: stop exploration; submit next; "
        "If final query allowed, run it once then submit.\n"
        "- If `submit_draft` says the conversation is terminated.\n"
        "- Plain text is invalid."
    )
