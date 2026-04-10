"""Model-generated naming variants for compiled tools."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from rl_task_foundry.config.models import DomainConfig, ModelRef, ProviderConfig
from rl_task_foundry.infra.json_chat_client import (
    JsonChatCompletionError,
    request_json_chat_completion,
)
from rl_task_foundry.schema.path_catalog import PathSpec
from rl_task_foundry.tools.models import ToolBundle, ToolSpec

_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,100}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolNameVariant(StrictModel):
    semantic_key: str
    name: str
    description: str


class ToolNameVariantResponse(StrictModel):
    variants: list[ToolNameVariant]


class ToolNamingGenerationError(RuntimeError):
    """Raised when model-generated naming cannot be validated."""


def _naming_prompt(
    *,
    domain: DomainConfig,
    path: PathSpec,
    tool_level: int,
    bundle: ToolBundle,
    task_question: str | None = None,
    question_family: str | None = None,
    outcome_type: str | None = None,
    feedback: str | None = None,
) -> list[dict[str, str]]:
    tool_payload = [
        {
            "semantic_key": tool.semantic_key,
            "kind": tool.kind,
            "current_name": tool.name,
            "description": tool.description,
            "parameters": [parameter.name for parameter in tool.parameters],
            "output_fields": tool.output_fields,
        }
        for tool in bundle.tools
    ]
    level_instruction = (
        "Use semi-indirect naming. Keep names discoverable but not fully literal. "
        "Do not restate the full table chain or path in the tool name. "
        "Use at most one direct schema noun from the path in each name, and prefer task wording "
        "over schema wording. Use compact names that sound like UI or workflow labels, not schema labels."
        if tool_level == 2
        else (
            "Use business-domain naming. Avoid exposing raw table or column identifiers directly "
            "when possible. Names should fit the task wording, not the schema wording."
        )
    )
    system_prompt = (
        "You rename compiled database tools without changing their semantics. "
        "Return JSON only. Preserve every semantic_key exactly once. "
        "All names must be unique snake_case identifiers. "
        "Descriptions must be short and concrete. "
        "Do not change parameters, output fields, or semantic meaning. "
        "Name the tools as they should appear in the final task package that the solver sees."
    )
    user_prompt = json.dumps(
        {
            "task": "rename_tools",
            "domain": {"name": domain.name, "language": domain.language},
            "target_tool_level": tool_level,
            "instruction": level_instruction,
            "quality_rules": {
                "preserve_semantics": True,
                "unique_names": True,
                "snake_case_only": True,
                "avoid_full_path_restatement": True,
                "prefer_question_words": True,
                "max_one_direct_schema_noun_per_name": True,
            },
            "path": {
                "path_id": path.path_id,
                "tables": path.tables,
                "hop_count": path.hop_count,
            },
            "task_context": {
                "question": task_question,
                "question_family": question_family,
                "outcome_type": outcome_type,
            },
            "tools": tool_payload,
            "response_schema": {
                "variants": [
                    {
                        "semantic_key": "same as input",
                        "name": "unique snake_case identifier",
                        "description": "short description",
                    }
                ]
            },
            "examples": _few_shot_examples(tool_level=tool_level),
            "revision_feedback": feedback,
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _few_shot_examples(*, tool_level: int) -> dict[str, object]:
    if tool_level == 1:
        return {
            "note": "Examples show direct canonical naming style.",
            "patterns": [
                {
                    "kind": "lookup",
                    "bad": "inspect_root_target_records",
                    "better": "get_target_for_root_via_link",
                },
                {
                    "kind": "list_related",
                    "bad": "show_related_items",
                    "better": "list_target_for_root_via_link",
                },
            ],
        }
    return {
        "note": (
            "Examples show style only. Do not copy the nouns literally. "
            "Keep the semantic action, but avoid schema-chain restatement."
        ),
        "patterns": [
            {
                "kind": "lookup",
                "bad": "inspect_root_link_target_records",
                "better": "inspect_profile",
            },
            {
                "kind": "list_related",
                "bad": "list_root_link_target_rows",
                "better": "list_options",
            },
            {
                "kind": "count",
                "bad": "count_root_target_rows",
                "better": "count_matches",
            },
            {
                "kind": "exists",
                "bad": "has_root_target_row",
                "better": "has_match",
            },
            {
                "kind": "aggregate",
                "bad": "sum_root_target_amount",
                "better": "summarize_total",
            },
            {
                "kind": "timeline",
                "bad": "timeline_root_target_events",
                "better": "review_timeline",
            },
        ],
    }


async def _request_tool_name_variants(
    *,
    provider: ProviderConfig,
    model_ref: ModelRef,
    domain: DomainConfig,
    path: PathSpec,
    tool_level: int,
    bundle: ToolBundle,
    task_question: str | None = None,
    question_family: str | None = None,
    outcome_type: str | None = None,
    feedback: str | None = None,
    temperature: float | None = None,
) -> ToolNameVariantResponse:
    try:
        response = await request_json_chat_completion(
            provider=provider,
            model_ref=model_ref,
            messages=_naming_prompt(
                domain=domain,
                path=path,
                tool_level=tool_level,
                bundle=bundle,
                task_question=task_question,
                question_family=question_family,
                outcome_type=outcome_type,
                feedback=feedback,
            ),
            temperature=temperature if temperature is not None else (1.0 if tool_level == 2 else 0.0),
        )
    except JsonChatCompletionError as exc:
        raise ToolNamingGenerationError(str(exc)) from exc
    try:
        return _parse_tool_name_variant_response(response["content"])
    except (ValidationError, ToolNamingGenerationError) as exc:
        raise ToolNamingGenerationError("Tool naming response did not match expected schema") from exc


async def generate_named_tool_bundle(
    *,
    provider: ProviderConfig,
    model_ref: ModelRef,
    domain: DomainConfig,
    path: PathSpec,
    bundle: ToolBundle,
    task_question: str | None = None,
    question_family: str | None = None,
    outcome_type: str | None = None,
    feedback: str | None = None,
    temperature: float | None = None,
) -> ToolBundle:
    """Generate model-based names/descriptions for an existing compiled tool bundle."""

    if bundle.tool_level == 1:
        return bundle

    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            response = await _request_tool_name_variants(
                provider=provider,
                model_ref=model_ref,
                domain=domain,
                path=path,
                tool_level=bundle.tool_level,
                bundle=bundle,
                task_question=task_question,
                question_family=question_family,
                outcome_type=outcome_type,
                feedback=feedback,
                temperature=temperature,
            )
            return _apply_name_variants(bundle, response)
        except ToolNamingGenerationError as exc:
            last_error = exc
    raise ToolNamingGenerationError("Tool naming generation failed after 3 attempts") from last_error


def _apply_name_variants(bundle: ToolBundle, response: ToolNameVariantResponse) -> ToolBundle:
    variants_by_key = {variant.semantic_key: variant for variant in response.variants}
    expected_keys = {tool.semantic_key for tool in bundle.tools}
    if set(variants_by_key) != expected_keys:
        raise ToolNamingGenerationError("Tool naming response semantic keys did not match bundle")

    seen_names: set[str] = set()
    updated_tools: list[ToolSpec] = []
    for tool in bundle.tools:
        variant = variants_by_key[tool.semantic_key]
        if not _TOOL_NAME_PATTERN.match(variant.name):
            raise ToolNamingGenerationError(f"Generated tool name is not snake_case: {variant.name}")
        if variant.name in seen_names:
            raise ToolNamingGenerationError(f"Generated duplicate tool name: {variant.name}")
        seen_names.add(variant.name)
        updated_tools.append(
            replace(
                tool,
                name=variant.name,
                description=variant.description.strip() or tool.description,
                name_source="model_generated",
            )
        )
    return ToolBundle(
        bundle_id=bundle.bundle_id,
        path_id=bundle.path_id,
        tool_level=bundle.tool_level,
        tools=updated_tools,
    )


def _parse_tool_name_variant_response(content: str) -> ToolNameVariantResponse:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ToolNamingGenerationError("Tool naming response was not valid JSON") from exc

    if isinstance(payload, dict) and "variants" not in payload and "tools" in payload:
        payload = {"variants": payload["tools"]}
    if isinstance(payload, dict) and isinstance(payload.get("variants"), list):
        payload = {
            "variants": [
                {
                    "semantic_key": variant.get("semantic_key"),
                    "name": variant.get("name"),
                    "description": variant.get("description"),
                }
                if isinstance(variant, dict)
                else variant
                for variant in payload["variants"]
            ]
        }
    return ToolNameVariantResponse.model_validate(payload)
