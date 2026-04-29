"""Tool-call budget prompt helper for the synthesis agent."""

from __future__ import annotations


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    return (
        "# Tool Budget\n"
        f"Use tools as needed; hard cap near {max_tool_calls} tool calls "
        "before `submit_draft`.\n"
        "- No per-stage data-tool quota.\n"
        "- Submit as soon as the final query evidence is ready. If a "
        "ToolBudgetFeedback reminder appears, stop exploration and call "
        "`submit_draft` unless one final label query is strictly needed.\n"
        "- If `submit_draft` says terminated, stop. Plain text is invalid."
    )
