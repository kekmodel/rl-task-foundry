"""Shared actor-facing rendered prompt builder for synthesized environments."""

from __future__ import annotations

import json

from rl_task_foundry.synthesis.contracts import TaskContract


def build_rendered_user_prompt(
    task: TaskContract,
    *,
    anchor_entity: dict[str, object] | None = None,
) -> str:
    """Render the actor-visible user prompt from the task contract and anchor."""

    entity_block = json.dumps(anchor_entity or {}, ensure_ascii=False, sort_keys=True)
    normalized_question = task.question.strip()
    if normalized_question.startswith("<entity>\n"):
        raise ValueError("TaskContract.question must not include an entity block")
    return (
        "<entity>\n"
        f"{entity_block}\n"
        "</entity>\n\n"
        f"{normalized_question}"
    )
