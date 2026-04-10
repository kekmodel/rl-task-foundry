"""Prompt builders for solver agents."""

from __future__ import annotations

from rl_task_foundry.tasks.models import TaskSpec


def build_solver_prompt(task: TaskSpec) -> str:
    """Create the base solver prompt."""

    field_list = ", ".join(field.name for field in task.answer_schema.fields)
    return (
        "You are a solver agent for a verifiable database task. "
        "Use only the provided tools, ground every answer in tool evidence, "
        "and submit your final answer with the submit_result tool when you are ready. "
        f"Respond in {task.language}. "
        f"Required fields: {field_list}. "
        "Do not invent fields. Do not answer in free-form text instead of calling submit_result."
    )
