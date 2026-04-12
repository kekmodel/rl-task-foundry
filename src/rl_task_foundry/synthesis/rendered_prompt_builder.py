"""Shared actor-facing rendered prompt builder for synthesized environments."""

from __future__ import annotations

import json
from typing import Any

from rl_task_foundry.synthesis.contracts import TaskContract
from rl_task_foundry.synthesis.schema_inference import extract_prompt_schema_from_canonical


def build_rendered_user_prompt(
    task: TaskContract,
    *,
    anchor_entity: dict[str, object] | None = None,
    canonical_answer: object | None = None,
    submit_schema: dict[str, Any] | None = None,
) -> str:
    """Render the actor-visible user prompt from the task contract and grounded label."""

    entity_block = json.dumps(anchor_entity or {}, ensure_ascii=False, sort_keys=True)
    if submit_schema is None:
        submit_schema = extract_prompt_schema_from_canonical(
            canonical_answer if canonical_answer is not None else {}
        )
    submit_schema_text = json.dumps(
        submit_schema,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (
        "<entity>\n"
        f"{entity_block}\n"
        "</entity>\n\n"
        f"{task.question.strip()}\n\n"
        "# Submit Result Format\n"
        f"{submit_schema_text}"
    )
