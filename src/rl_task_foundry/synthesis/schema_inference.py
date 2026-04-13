"""Schema inference helpers derived directly from canonical answers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import TypeAdapter, create_model

from rl_task_foundry.synthesis.contracts import (
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
)


def extract_output_schema_from_canonical(canonical_answer: object) -> OutputSchemaContract:
    """Infer the internal output-schema contract from a grounded canonical answer."""

    root = _infer_output_field("answer", canonical_answer)
    primary_output_format = "json_array" if root.type == OutputFieldType.LIST else "json_object"
    return OutputSchemaContract(root=root, primary_output_format=primary_output_format)


def extract_prompt_schema_from_canonical(canonical_answer: object) -> dict[str, Any]:
    """Infer a JSON Schema payload for the solver-facing rendered prompt."""

    annotation = _infer_python_annotation("AnswerSchema", canonical_answer)
    return TypeAdapter(annotation).json_schema()


def _infer_output_field(name: str, value: object) -> OutputFieldContract:
    if isinstance(value, dict):
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.OBJECT,
            fields=[
                _infer_output_field(str(child_name), child_value)
                for child_name, child_value in value.items()
            ],
        )
    if isinstance(value, list):
        sample = value[0] if value else ""
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.LIST,
            ordered=True,
            items=_infer_output_field("item", sample),
        )
    if isinstance(value, bool):
        return OutputFieldContract(name=name, type=OutputFieldType.BOOL)
    if isinstance(value, int) and not isinstance(value, bool):
        return OutputFieldContract(name=name, type=OutputFieldType.INT)
    if isinstance(value, float):
        return OutputFieldContract(name=name, type=OutputFieldType.FLOAT)
    if isinstance(value, datetime):
        return OutputFieldContract(name=name, type=OutputFieldType.DATETIME)
    if isinstance(value, date):
        return OutputFieldContract(name=name, type=OutputFieldType.DATE)
    return OutputFieldContract(name=name, type=OutputFieldType.STRING)


def _infer_python_annotation(model_name: str, value: object) -> Any:
    if isinstance(value, dict):
        fields = {
            str(key): (_infer_python_annotation(f"{model_name}_{key}", child), ...)
            for key, child in value.items()
        }
        return create_model(model_name, **fields)  # type: ignore[call-overload]
    if isinstance(value, list):
        item_value = value[0] if value else ""
        item_annotation = _infer_python_annotation(f"{model_name}Item", item_value)
        return list[item_annotation]
    if isinstance(value, bool):
        return bool
    if isinstance(value, int) and not isinstance(value, bool):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, datetime):
        return datetime
    if isinstance(value, date):
        return date
    return str
