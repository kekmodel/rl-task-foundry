"""Shared actor-facing rendered prompt builder for synthesized environments."""

from __future__ import annotations

import json

from rl_task_foundry.synthesis.contracts import (
    OutputFieldContract,
    OutputFieldType,
    TaskContract,
)


def build_rendered_user_prompt(task: TaskContract) -> str:
    """Render the actor-visible user prompt from the task contract."""

    output_schema_text = json.dumps(
        build_output_schema_prompt_payload(task.output_schema.root),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    lines = [task.question.strip()]
    if task.constraint_summary:
        lines.extend(["", "Constraints:"])
        lines.extend(f"- {item.summary}" for item in task.constraint_summary)
    if task.instance_parameters:
        lines.extend(["", "Instance Parameters:"])
        lines.extend(
            f"- {key}: {json.dumps(value, ensure_ascii=False)}"
            for key, value in task.instance_parameters.items()
        )
    prompt_body = "\n".join(lines).rstrip()
    submit_result_block = "\n".join(
        [
            "# Submit Result Format",
            "Submit your final answer via submit_result as a JSON string matching this schema.",
            output_schema_text,
        ]
    )
    return f"{prompt_body}\n\n{submit_result_block}"


def build_output_schema_prompt_payload(field: OutputFieldContract) -> dict[str, object]:
    """Render the actor-facing JSON-serializable prompt payload for an output field."""

    payload: dict[str, object] = {"type": field.type.value}
    if field.description:
        payload["description"] = field.description
    if field.nullable:
        payload["nullable"] = True
    if field.type == OutputFieldType.ENUM and field.enum_values:
        payload["enum"] = list(field.enum_values)
    if field.type == OutputFieldType.OBJECT:
        payload["properties"] = {
            child.name: build_output_schema_prompt_payload(child) for child in field.fields
        }
        payload["required"] = [child.name for child in field.fields if not child.nullable]
    elif field.type == OutputFieldType.LIST and field.items is not None:
        payload["ordered"] = field.ordered
        if field.sort_key is not None:
            payload["sort_key"] = list(field.sort_key)
        if field.unique_elements:
            payload["unique_elements"] = True
        payload["items"] = build_output_schema_prompt_payload(field.items)
    return payload
