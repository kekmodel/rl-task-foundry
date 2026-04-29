"""Shared actor-facing rendered prompt builder for synthesized environments."""

from __future__ import annotations

import json
from typing import Any

from rl_task_foundry.synthesis.contracts import TaskContract


def build_rendered_user_prompt(
    task: TaskContract,
    *,
    anchor_entity: dict[str, object] | None = None,
    canonical_answer: object | None = None,
    submit_schema: dict[str, Any] | None = None,
) -> str:
    """Render the actor-visible user prompt from the task contract and grounded label."""

    del canonical_answer, submit_schema
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
