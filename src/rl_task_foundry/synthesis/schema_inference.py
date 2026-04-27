"""Schema inference helpers derived directly from canonical answers."""

from __future__ import annotations

from datetime import date, datetime

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


def extract_prompt_schema_from_canonical(canonical_answer: object) -> dict[str, object]:
    """Infer a JSON Schema payload for the solver-facing rendered prompt."""

    annotation = _python_annotation_from_output_field(
        "AnswerSchema",
        extract_output_schema_from_canonical(canonical_answer).root,
    )
    return TypeAdapter(annotation).json_schema()


def _infer_output_field(name: str, value: object) -> OutputFieldContract:
    if isinstance(value, list):
        return _infer_list_output_field(
            name,
            [value],
            nullable=False,
            force_length=len(value),
        )
    return _infer_output_field_from_values(name, [value])


def _infer_output_field_from_values(
    name: str,
    values: list[object],
) -> OutputFieldContract:
    nullable = any(value is None for value in values)
    non_null_values = [value for value in values if value is not None]
    if not non_null_values:
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.STRING,
            nullable=True,
        )

    if all(isinstance(value, dict) for value in non_null_values):
        objects = [value for value in non_null_values if isinstance(value, dict)]
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.OBJECT,
            nullable=nullable,
            fields=[
                _infer_output_field_from_values(
                    str(child_name),
                    [
                        item.get(child_name) if child_name in item else None
                        for item in objects
                    ],
                )
                for child_name in _ordered_object_keys(objects)
            ],
        )

    if all(isinstance(value, list) for value in non_null_values):
        return _infer_list_output_field(
            name,
            [value for value in non_null_values if isinstance(value, list)],
            nullable=nullable,
        )

    if all(isinstance(value, bool) for value in non_null_values):
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.BOOL,
            nullable=nullable,
        )
    if all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in non_null_values
    ):
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.INT,
            nullable=nullable,
        )
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in non_null_values
    ):
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.FLOAT,
            nullable=nullable,
        )
    if all(isinstance(value, datetime) for value in non_null_values):
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.DATETIME,
            nullable=nullable,
        )
    if all(
        isinstance(value, date) and not isinstance(value, datetime)
        for value in non_null_values
    ):
        return OutputFieldContract(
            name=name,
            type=OutputFieldType.DATE,
            nullable=nullable,
        )
    return OutputFieldContract(
        name=name,
        type=OutputFieldType.STRING,
        nullable=nullable,
    )


def _infer_list_output_field(
    name: str,
    values: list[list[object]],
    *,
    nullable: bool,
    force_length: int | None = None,
) -> OutputFieldContract:
    item_values = [item for value in values for item in value]
    item = _infer_output_field_from_values("item", item_values or [""])
    if force_length is not None:
        length = force_length
    else:
        lengths = {len(value) for value in values}
        length = lengths.pop() if len(lengths) == 1 else None
    return OutputFieldContract(
        name=name,
        type=OutputFieldType.LIST,
        nullable=nullable,
        ordered=True,
        length=length,
        items=item,
    )


def _ordered_object_keys(objects: list[dict[object, object]]) -> list[object]:
    keys: list[object] = []
    seen: set[object] = set()
    for item in objects:
        for key in item:
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def _python_annotation_from_output_field(
    model_name: str,
    field: OutputFieldContract,
) -> object:
    if field.type is OutputFieldType.OBJECT:
        fields = {
            child.name: (
                _python_annotation_from_output_field(
                    f"{model_name}_{child.name}",
                    child,
                ),
                ...,
            )
            for child in field.fields
        }
        annotation: object = create_model(
            model_name,
            **fields,
        )  # type: ignore[call-overload]
    elif field.type is OutputFieldType.LIST:
        item_annotation = (
            _python_annotation_from_output_field(f"{model_name}Item", field.items)
            if field.items is not None
            else str
        )
        annotation = list[item_annotation]  # type: ignore[valid-type]
    elif field.type is OutputFieldType.BOOL:
        annotation = bool
    elif field.type is OutputFieldType.INT:
        annotation = int
    elif field.type is OutputFieldType.FLOAT:
        annotation = float
    elif field.type is OutputFieldType.DATETIME:
        annotation = datetime
    elif field.type is OutputFieldType.DATE:
        annotation = date
    else:
        annotation = str
    if field.nullable:
        return annotation | None
    return annotation
