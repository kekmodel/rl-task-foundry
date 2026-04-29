"""Tool-call budget prompt helper for the synthesis agent."""

from __future__ import annotations

FIRST_SUBMIT_MAX_DATA_TOOLS = 3
FEEDBACK_REPAIR_MAX_DATA_TOOLS = 3


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    first_submit_deadline = max(1, max_tool_calls // 3)
    return (
        "# Draft Submission Budget\n"
        f"Budget: {max_tool_calls} tool calls total incl. data tools and "
        "`submit_draft`.\n"
        f"- Call `submit_draft` within first {first_submit_deadline} "
        "tool calls; "
        f"max {FIRST_SUBMIT_MAX_DATA_TOOLS} data tools before first "
        "`submit_draft`.\n"
        f"- After feedback, max {FEEDBACK_REPAIR_MAX_DATA_TOOLS} data tools; "
        "binding feedback uses none. At ToolBudgetFeedback, submit next; "
        "only label `query` if rows missing.\n"
        "- If `submit_draft` says the conversation is terminated, stop.\n"
        "- Plain text is invalid."
    )
