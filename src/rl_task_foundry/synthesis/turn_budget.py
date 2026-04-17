"""Tool-call budget prompt helper for the synthesis agent.

The synthesis agent is the only place we inject budget guidance. The solver is
intentionally left uncoached so its pass-rate stays an unbiased measurement of
task difficulty.
"""

from __future__ import annotations


def build_tool_call_budget_instruction(*, max_tool_calls: int) -> str:
    first_submit_deadline = max(1, max_tool_calls // 3)
    return (
        "# Commit Rule\n"
        f"You have a budget of {max_tool_calls} tool calls. Every tool call "
        "counts the same: atomic calls (get/find/calc/rank) and submit_draft "
        "alike.\n"
        "\n"
        f"- Within your first {first_submit_deadline} tool calls, "
        "submit_draft must have been called at least once.\n"
        "- Spend at most 3 atomic calls before your first submit.\n"
        "- A too_easy or too_hard rejection is feedback, not failure. Keep "
        "the previous draft and resubmit with ONE change.\n"
        f"- Exhausting the {max_tool_calls}-call budget without a submit "
        "discards the trial.\n"
        "\n"
        "Submit early, iterate on rejection. Exploration is cheap after the "
        "first submit; exploration without a submit is waste."
    )
