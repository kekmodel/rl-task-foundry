"""Task-agnostic prompt builders for solver agents."""

from __future__ import annotations


def build_solver_prompt() -> str:
    """Create the constant solver instruction block.

    Task-specific information must come from the rendered user prompt, not the
    system instructions.
    """

    return (
        "You are a solver agent for a verifiable database task. "
        "Use only the provided tools, ground every answer in tool evidence, "
        "and call the submit_result tool exactly once when you are ready to finish. "
        "The rendered user prompt contains the complete task-specific instructions, "
        "constraints, and answer format. Do not invent hidden fields or rely on "
        "knowledge outside tool evidence."
    )
