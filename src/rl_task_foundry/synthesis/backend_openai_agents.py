"""OpenAI Agents SDK-backed synthesis runtime backend."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from rl_task_foundry.config.models import ModelRef, ProviderConfig, SynthesisRuntimeConfig
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_phase_input,
    build_synthesis_phase_instructions,
)
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, ConstraintKind
from rl_task_foundry.synthesis.tool_runtime import ToolExecutor
from rl_task_foundry.synthesis.runtime import (
    ArtifactGenerationOutput,
    CategoryInferenceOutput,
    LabelConstructionOutput,
    SchemaExplorationOutput,
    SynthesisMemoryEntry,
    SynthesisPhase,
    SynthesisStageRequest,
    SynthesisStageResult,
    TaskSynthesisOutput,
    SynthesisToolTraceEntry,
)


def _load_sdk_components() -> SimpleNamespace:
    from agents import (
        Agent,
        AgentOutputSchema,
        ModelSettings,
        OpenAIChatCompletionsModel,
        Runner,
        SQLiteSession,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    return SimpleNamespace(
        Agent=Agent,
        AgentOutputSchema=AgentOutputSchema,
        AsyncOpenAI=AsyncOpenAI,
        ModelSettings=ModelSettings,
        OpenAIChatCompletionsModel=OpenAIChatCompletionsModel,
        Runner=Runner,
        SQLiteSession=SQLiteSession,
        set_tracing_disabled=set_tracing_disabled,
    )


def _trace_stub(kind: str, request: SynthesisStageRequest, provider: str, model: str) -> str:
    return f"memory://{kind}/{request.db_id}/{request.phase.value}/{provider}/{model}"


def _extract_token_usage(run_result: Any) -> dict[str, int]:
    usage = getattr(getattr(run_result, "context_wrapper", None), "usage", None)
    if usage is None:
        return {}
    return {
        "requests": int(getattr(usage, "requests", 0) or 0),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _extract_turn_count(run_result: Any) -> int:
    explicit_turn_count = getattr(run_result, "_current_turn", None)
    if explicit_turn_count is None:
        explicit_turn_count = getattr(run_result, "current_turn", None)
    if explicit_turn_count:
        return int(explicit_turn_count)
    raw_responses = getattr(run_result, "raw_responses", None)
    if isinstance(raw_responses, list) and raw_responses:
        return len(raw_responses)
    usage = getattr(getattr(run_result, "context_wrapper", None), "usage", None)
    requests = getattr(usage, "requests", 0) if usage is not None else 0
    return int(requests or 0)


def _extract_tool_call_name(item: Any) -> str | None:
    for attr in ("name", "tool_name"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return value
    raw_item = getattr(item, "raw_item", None)
    if raw_item is not None:
        for attr in ("name", "tool_name"):
            value = getattr(raw_item, attr, None)
            if isinstance(value, str) and value:
                return value
    if isinstance(item, str) and item.startswith("tool-call(") and item.endswith(")"):
        return item[len("tool-call(") : -1]
    return None


def _extract_tool_traces(
    run_result: Any,
    *,
    request: SynthesisStageRequest,
    provider: str,
    model: str,
) -> list[SynthesisToolTraceEntry]:
    traces: list[SynthesisToolTraceEntry] = []
    for item in getattr(run_result, "new_items", []) or []:
        tool_name = _extract_tool_call_name(item)
        if tool_name is None:
            continue
        traces.append(
            SynthesisToolTraceEntry(
                phase=request.phase,
                provider=provider,
                model=model,
                tool_name=tool_name,
                raw_repr=repr(item),
            )
        )
    return traces


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _phase_output_type(phase: SynthesisPhase) -> type[BaseModel]:
    if phase == SynthesisPhase.SCHEMA_EXPLORATION:
        return SchemaExplorationOutput
    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        return CategoryInferenceOutput
    if phase == SynthesisPhase.LABEL_CONSTRUCTION:
        return LabelConstructionOutput
    if phase == SynthesisPhase.TASK_SYNTHESIS:
        return TaskSynthesisOutput
    return ArtifactGenerationOutput


def _normalize_tool_definition(definition: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(definition, dict):
        raise TypeError("tool definitions must be dict-like payloads")
    return {
        "name": definition["name"],
        "description": definition["description"],
        "params_schema": dict(definition.get("params_schema", {})),
        "semantic_key": definition.get("semantic_key"),
    }


def _make_sdk_tool(definition: dict[str, Any], executor: ToolExecutor) -> object:
    from agents import FunctionTool
    from agents.strict_schema import ensure_strict_json_schema

    params_json_schema = ensure_strict_json_schema(dict(definition["params_schema"]))

    async def _invoke_tool(_tool_context: Any, input_json: str) -> Any:
        payload = json.loads(input_json) if input_json else {}
        if not isinstance(payload, dict):
            raise ValueError("Tool input must be a JSON object")
        result = executor(payload)
        if hasattr(result, "__await__"):
            return await result
        return result

    return FunctionTool(
        name=str(definition["name"]),
        description=str(definition["description"]),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=True,
    )


def _build_agent(
    sdk: SimpleNamespace,
    *,
    request: SynthesisStageRequest,
    model: Any,
    structured_output: bool,
    tools: list[object] | None = None,
) -> Any:
    output_type: Any | None = None
    if structured_output:
        output_type = _phase_output_type(request.phase)
        if request.phase in {
            SynthesisPhase.LABEL_CONSTRUCTION,
            SynthesisPhase.TASK_SYNTHESIS,
            SynthesisPhase.ARTIFACT_GENERATION,
        }:
            output_schema_factory = getattr(sdk, "AgentOutputSchema", None)
            if output_schema_factory is not None:
                output_type = output_schema_factory(output_type, strict_json_schema=False)
    model_settings_kwargs: dict[str, Any] = {"parallel_tool_calls": False}
    if request.phase == SynthesisPhase.SCHEMA_EXPLORATION and tools:
        model_settings_kwargs["tool_choice"] = "required"
    return sdk.Agent(
        name=f"synthesis-{request.phase.value}",
        instructions=build_synthesis_phase_instructions(request.phase),
        model=model,
        tools=tools or [],
        output_type=output_type,
        model_settings=sdk.ModelSettings(**model_settings_kwargs),
    )


def _phase_max_turns(
    request: SynthesisStageRequest,
    *,
    runtime_max_turns: int,
    tools: list[object],
) -> int:
    if request.phase == SynthesisPhase.SCHEMA_EXPLORATION and tools:
        return max(runtime_max_turns, 12)
    return runtime_max_turns


def _normalize_category_value(value: Any) -> str | None:
    if isinstance(value, CategoryTaxonomy):
        return value.value
    if not isinstance(value, str):
        return None
    try:
        return CategoryTaxonomy(value).value
    except ValueError:
        return None


def _normalize_fact_value_type(value: Any) -> tuple[str | None, bool]:
    if not isinstance(value, str):
        return None, False
    normalized = value.strip().lower()
    mapping = {
        "str": "str",
        "string": "str",
        "int": "int",
        "integer": "int",
        "float": "float",
        "number": "float",
        "bool": "bool",
        "boolean": "bool",
        "date": "date",
        "datetime": "datetime",
        "list[str]": "list[str]",
        "list[int]": "list[int]",
        "list[float]": "list[float]",
    }
    if normalized in mapping:
        return mapping[normalized], False
    nullable_prefixes = ("nullable_", "optional_")
    for prefix in nullable_prefixes:
        if normalized.startswith(prefix):
            base = mapping.get(normalized[len(prefix) :])
            return base, base is not None
    if normalized.endswith("_or_null"):
        base = mapping.get(normalized[: -len("_or_null")])
        return base, base is not None
    if normalized.endswith("|null"):
        base = mapping.get(normalized[: -len("|null")])
        return base, base is not None
    return None, False


def _infer_fact_value_type_from_key(key: str) -> str:
    normalized = key.strip().lower()
    if (
        normalized.endswith("_id")
        or normalized.endswith("_count")
        or normalized.endswith("_index")
        or normalized.endswith("_position")
        or normalized.endswith("_year")
        or normalized.endswith("_month")
        or normalized.endswith("_day")
    ):
        return "int"
    if (
        normalized.startswith("is_")
        or normalized.startswith("has_")
        or normalized.startswith("can_")
        or normalized.endswith("_flag")
        or normalized.endswith("_enabled")
        or normalized.endswith("_active")
    ):
        return "bool"
    if (
        normalized.endswith("_amount")
        or normalized.endswith("_price")
        or normalized.endswith("_cost")
        or normalized.endswith("_rate")
        or normalized.endswith("_ratio")
        or normalized.endswith("_score")
        or normalized.endswith("_total")
    ):
        return "float"
    if normalized.endswith("_date") or normalized == "date":
        return "date"
    if (
        normalized.endswith("_at")
        or normalized.endswith("_time")
        or normalized.endswith("_timestamp")
        or normalized.endswith("_datetime")
    ):
        return "datetime"
    return "str"


def _normalize_fact_schema_entry(fact: object) -> tuple[object, bool, str | None]:
    if isinstance(fact, str) and fact.strip():
        fact_name = fact.strip()
        return (
            {
                "key": fact_name,
                "entity_ref": "answer",
                "attribute": fact_name,
                "value_type": _infer_fact_value_type_from_key(fact_name),
            },
            True,
            "facts_schema_key_list_normalized",
        )
    if not isinstance(fact, dict):
        return fact, False, None
    if {"key", "entity_ref", "attribute", "value_type"} <= set(fact):
        normalized_value_type, inferred_nullable = _normalize_fact_value_type(
            fact.get("value_type")
        )
        if normalized_value_type is None:
            return fact, False, None
        updated_fact = dict(fact)
        changed = False
        if updated_fact.get("value_type") != normalized_value_type:
            updated_fact["value_type"] = normalized_value_type
            changed = True
        if inferred_nullable and updated_fact.get("nullable") is not True:
            updated_fact["nullable"] = True
            changed = True
        if not changed:
            return fact, False, None
        return updated_fact, True, "facts_schema_nullable_alias_normalized"
    fact_name = fact.get("name")
    fact_value_type, inferred_nullable = _normalize_fact_value_type(fact.get("type"))
    if isinstance(fact_name, str) and fact_value_type is not None:
        normalized_fact = {
            "key": fact_name,
            "entity_ref": "answer",
            "attribute": fact_name,
            "value_type": fact_value_type,
        }
        if inferred_nullable:
            normalized_fact["nullable"] = True
        return (
            normalized_fact,
            True,
            "facts_schema_normalized",
        )
    return fact, False, None


def _normalize_constraint_kind(value: object) -> tuple[str | None, bool]:
    if isinstance(value, ConstraintKind):
        return value.value, False
    if not isinstance(value, str):
        return None, False
    normalized = value.strip().lower()
    try:
        return ConstraintKind(normalized).value, False
    except ValueError:
        pass
    alias_mapping = {
        "relationship_traversal": ConstraintKind.OTHER.value,
        "join_depth": ConstraintKind.OTHER.value,
        "fk_traversal": ConstraintKind.OTHER.value,
        "foreign_key_traversal": ConstraintKind.OTHER.value,
        "tie_break": ConstraintKind.UNIQUENESS.value,
    }
    mapped = alias_mapping.get(normalized)
    if mapped is not None:
        return mapped, True
    return None, False


def _repair_split_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    try:
        prefix, index = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    remainder = text[index:].strip()
    if not remainder.startswith(",") or not isinstance(prefix, dict):
        return None
    try:
        suffix = json.loads("{" + remainder[1:])
    except json.JSONDecodeError:
        return None
    if not isinstance(suffix, dict):
        return None
    merged = dict(prefix)
    for key, value in suffix.items():
        merged.setdefault(key, value)
    return merged


def _dedupe_repair_codes(repair_codes: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for code in repair_codes:
        if code in seen:
            continue
        seen.add(code)
        deduped.append(code)
    return deduped


def _normalize_phase_payload(
    request: SynthesisStageRequest,
    final_output: Any,
) -> tuple[BaseModel, list[str]]:
    output_type = _phase_output_type(request.phase)
    if isinstance(final_output, output_type):
        return final_output, []
    if isinstance(final_output, BaseModel):
        return output_type.model_validate(final_output.model_dump(mode="python")), []
    if isinstance(final_output, dict):
        final_output, repair_codes = _coerce_phase_payload_dict(request, final_output)
        return output_type.model_validate(final_output), _dedupe_repair_codes(repair_codes)
    if isinstance(final_output, str):
        payload_text = _extract_json_object_text(final_output)
        repair_codes: list[str] = []
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            repaired = _repair_split_json_object(payload_text)
            if repaired is None:
                raise
            repair_codes.append("split_json_repaired")
            parsed = repaired
        if not isinstance(parsed, dict):
            raise RuntimeError("synthesis backend expected a JSON object payload")
        parsed, coerced_codes = _coerce_phase_payload_dict(request, parsed)
        repair_codes.extend(coerced_codes)
        return output_type.model_validate(parsed), _dedupe_repair_codes(repair_codes)
    raise RuntimeError("synthesis backend returned an unsupported final_output type")


def _coerce_phase_payload_dict(
    request: SynthesisStageRequest,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    repair_codes: list[str] = []
    if request.phase == SynthesisPhase.ARTIFACT_GENERATION:
        proposed_environment = payload.get("proposed_environment")
        if isinstance(proposed_environment, dict):
            normalized_proposed_environment = dict(proposed_environment)
            promoted_keys = ("task", "solution", "verifier", "shadow_verifier", "instance_space")
            for key in promoted_keys:
                if key in payload and key not in normalized_proposed_environment:
                    normalized_proposed_environment[key] = payload[key]
                    repair_codes.append("artifact_proposed_environment_nested")
            for verifier_key in ("verifier", "shadow_verifier"):
                verifier_payload = normalized_proposed_environment.get(verifier_key)
                if not isinstance(verifier_payload, dict):
                    continue
                facts_schema = verifier_payload.get("facts_schema")
                if not isinstance(facts_schema, dict):
                    continue
                facts = facts_schema.get("facts")
                if not isinstance(facts, list):
                    continue
                normalized_facts: list[object] = []
                changed = False
                for fact in facts:
                    normalized_fact, fact_changed, repair_code = _normalize_fact_schema_entry(fact)
                    normalized_facts.append(normalized_fact)
                    if fact_changed:
                        changed = True
                    if repair_code is not None:
                        repair_codes.append(repair_code)
                if changed:
                    updated_verifier = dict(verifier_payload)
                    updated_facts_schema = dict(facts_schema)
                    updated_facts_schema["facts"] = normalized_facts
                    updated_verifier["facts_schema"] = updated_facts_schema
                    normalized_proposed_environment[verifier_key] = updated_verifier
                    repair_codes.append("facts_schema_normalized")
            task_payload = normalized_proposed_environment.get("task")
            if isinstance(task_payload, dict):
                task_updates: dict[str, Any] = {}
                label_output = request.previous_outputs.get(SynthesisPhase.LABEL_CONSTRUCTION)
                task_output = request.previous_outputs.get(SynthesisPhase.TASK_SYNTHESIS)
                if isinstance(label_output, LabelConstructionOutput):
                    if "output_schema" not in task_payload:
                        task_updates["output_schema"] = label_output.output_schema.model_dump(
                            mode="json"
                        )
                    if "difficulty_vector" not in task_payload:
                        task_updates["difficulty_vector"] = label_output.difficulty_vector.model_dump(
                            mode="json"
                        )
                    if "instance_parameters" not in task_payload:
                        task_updates["instance_parameters"] = label_output.instance_parameters
                if isinstance(task_output, TaskSynthesisOutput):
                    if "question" not in task_payload:
                        task_updates["question"] = task_output.question
                    if "constraint_summary" not in task_payload:
                        task_updates["constraint_summary"] = [
                            item.model_dump(mode="json") for item in task_output.constraint_summary
                        ]
                if "category" not in task_payload and request.requested_category is not None:
                    task_updates["category"] = request.requested_category.value
                if task_updates:
                    updated_task = dict(task_payload)
                    updated_task.update(task_updates)
                    normalized_proposed_environment["task"] = updated_task
                    task_payload = updated_task
                    repair_codes.append("artifact_task_completed_from_previous_outputs")
                constraint_summary = task_payload.get("constraint_summary")
                if isinstance(constraint_summary, list):
                    normalized_constraints: list[object] = []
                    changed = False
                    for item in constraint_summary:
                        if not isinstance(item, dict):
                            normalized_constraints.append(item)
                            continue
                        normalized_kind, kind_changed = _normalize_constraint_kind(item.get("kind"))
                        if normalized_kind is None:
                            normalized_constraints.append(item)
                            continue
                        if kind_changed or item.get("kind") != normalized_kind:
                            updated_item = dict(item)
                            updated_item["kind"] = normalized_kind
                            normalized_constraints.append(updated_item)
                            changed = True
                            repair_codes.append("constraint_kind_normalized")
                        else:
                            normalized_constraints.append(item)
                    if changed:
                        updated_task = dict(task_payload)
                        updated_task["constraint_summary"] = normalized_constraints
                        normalized_proposed_environment["task"] = updated_task
            task_output = request.previous_outputs.get(SynthesisPhase.TASK_SYNTHESIS)
            if (
                "instance_space" not in normalized_proposed_environment
                and isinstance(task_output, TaskSynthesisOutput)
            ):
                normalized_proposed_environment["instance_space"] = task_output.instance_space.model_dump(
                    mode="json"
                )
                repair_codes.append("artifact_instance_space_completed_from_previous_outputs")
            normalized_payload = dict(payload)
            normalized_payload["proposed_environment"] = normalized_proposed_environment
            pruned_top_level_fields: list[str] = []
            for key in promoted_keys:
                if key in normalized_payload:
                    del normalized_payload[key]
                    pruned_top_level_fields.append(key)
            if pruned_top_level_fields:
                repair_codes.append("artifact_top_level_fields_pruned")
            legacy_artifacts = normalized_payload.get("artifacts")
            if isinstance(legacy_artifacts, dict):
                normalized_artifacts = dict(legacy_artifacts)
                for legacy_name, contract_name in (
                    ("solution.py", "solution_source"),
                    ("verifier.py", "verifier_source"),
                    ("shadow_verifier.py", "shadow_verifier_source"),
                ):
                    if contract_name not in normalized_artifacts and legacy_name in normalized_artifacts:
                        normalized_artifacts[contract_name] = normalized_artifacts[legacy_name]
                        repair_codes.append("artifact_key_remapped")
                normalized_payload["artifacts"] = normalized_artifacts
            return normalized_payload, _dedupe_repair_codes(repair_codes)
        return payload, []

    phase = request.phase
    if phase == SynthesisPhase.LABEL_CONSTRUCTION:
        if {
            "canonical_answer_json",
            "output_schema",
            "difficulty_vector",
            "label_summary",
        } <= set(payload):
            return payload, []
        canonical_answer_json: str | None = None
        for key in ("canonical_answer_json", "answer_json"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                canonical_answer_json = value.strip()
                break
        if canonical_answer_json is None and "canonical_answer" in payload:
            canonical_answer_json = json.dumps(
                payload["canonical_answer"],
                ensure_ascii=False,
                sort_keys=True,
            )
            repair_codes.append("label_construction_answer_jsonized")
        remapped = {
            "canonical_answer_json": canonical_answer_json,
            "output_schema": payload.get("output_schema"),
            "difficulty_vector": payload.get("difficulty_vector")
            or payload.get("difficulty")
            or {},
            "instance_parameters": payload.get("instance_parameters") or payload.get("params") or {},
            "label_summary": payload.get("label_summary")
            or payload.get("rationale")
            or payload.get("uniqueness_strategy")
            or "label construction completed",
            "memory_summary": payload.get("memory_summary", "label construction completed"),
        }
        if canonical_answer_json is not None:
            repair_codes.append("label_construction_remapped")
        return remapped, repair_codes

    if phase == SynthesisPhase.TASK_SYNTHESIS:
        if {"question", "constraint_summary", "instance_space"} <= set(payload):
            return payload, []
        instance_space = payload.get("instance_space")
        if instance_space is None and "anchor_query" in payload:
            instance_space = {
                "anchor_query": payload.get("anchor_query"),
                "parameters": payload.get("parameters") or {},
                "sampling": payload.get("sampling") or {"strategy": "deterministic_hash", "seed": 0},
                "instance_count": payload.get("instance_count"),
            }
            repair_codes.append("task_synthesis_instance_space_nested")
        remapped = {
            "question": payload.get("question") or payload.get("user_prompt") or "",
            "constraint_summary": payload.get("constraint_summary") or payload.get("constraints") or [],
            "instance_space": instance_space,
            "memory_summary": payload.get("memory_summary", "task synthesis completed"),
        }
        if remapped["question"] or remapped["instance_space"] is not None:
            repair_codes.append("task_synthesis_remapped")
        return remapped, repair_codes

    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        if "selected_category" in payload and "rationale" in payload:
            return payload, []
        selected_category = _normalize_category_value(payload.get("category"))
        remapped = {
            "selected_category": selected_category,
            "rationale": payload.get("validation_notes")
            or payload.get("unique_answer_strategy")
            or payload.get("recommended_task_type")
            or "category inference completed",
            "memory_summary": payload.get("memory_summary", "category inference completed"),
        }
        if remapped["selected_category"] is not None:
            repair_codes.append("category_inference_remapped")
        return remapped, repair_codes

    if phase != SynthesisPhase.SCHEMA_EXPLORATION:
        return payload, []
    if {
        "domain_hypothesis",
        "candidate_categories",
        "sample_observations",
    } <= set(payload):
        return payload, []

    candidate_categories: list[str] = []
    sample_observations: list[str] = []
    explicit_category = _normalize_category_value(payload.get("category"))
    if explicit_category is not None:
        candidate_categories.append(explicit_category)
        repair_codes.append("schema_exploration_category_promoted")
    normalized_task_category = _normalize_category_value(payload.get("task_category"))
    if normalized_task_category is not None and normalized_task_category not in candidate_categories:
        candidate_categories.append(normalized_task_category)
    raw_categories = payload.get("categories")
    if isinstance(raw_categories, list):
        for item in raw_categories:
            candidate_name = item.get("name") if isinstance(item, dict) else item
            normalized_candidate = _normalize_category_value(candidate_name)
            if normalized_candidate is None or normalized_candidate in candidate_categories:
                continue
            candidate_categories.append(normalized_candidate)

    raw_observations = payload.get("sample_observations") or payload.get("observations") or payload.get("sample_rows")
    if isinstance(raw_observations, list):
        for item in raw_observations:
            if isinstance(item, str) and item.strip():
                sample_observations.append(item.strip())
                continue
            if isinstance(item, dict):
                insight = item.get("insight") or item.get("summary")
                tool_name = item.get("tool_name") or item.get("tool")
                result_preview = item.get("result_preview_json") or item.get("result_preview")
                parts = [
                    part
                    for part in (
                        f"tool={tool_name}" if isinstance(tool_name, str) and tool_name else None,
                        insight if isinstance(insight, str) and insight else None,
                        (
                            f"result={result_preview}"
                            if isinstance(result_preview, str) and result_preview
                            else None
                        ),
                    )
                    if part is not None
                ]
                if parts:
                    sample_observations.append("; ".join(parts))

    remapped = {
        "domain_hypothesis": payload.get("domain_fit")
        or payload.get("compositionality")
        or payload.get("reasoning")
        or "schema exploration completed",
        "candidate_categories": candidate_categories,
        "sample_observations": sample_observations,
        "memory_summary": payload.get("memory_summary", "schema exploration completed"),
    }
    if (
        remapped["domain_hypothesis"] != "schema exploration completed"
        or candidate_categories
        or sample_observations
    ):
        repair_codes.append("schema_exploration_remapped")
    return remapped, repair_codes


@dataclass(slots=True)
class OpenAIAgentsSynthesisBackend:
    model_ref: ModelRef
    provider_config: ProviderConfig
    runtime_config: SynthesisRuntimeConfig
    session_db_path: Path | None = None
    traces_dir: Path | None = None
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    tool_executors: dict[str, ToolExecutor] = field(default_factory=dict)

    @property
    def provider_name(self) -> str:
        return self.model_ref.provider

    @property
    def model_name(self) -> str:
        return self.model_ref.model

    def _resolve_api_key(self) -> str:
        env_name = self.provider_config.api_key_env
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
        if self.provider_config.type == "openai_compatible":
            return "dummy"
        raise RuntimeError(f"Required API key env var is missing: {env_name}")

    def _build_model(self, sdk: SimpleNamespace) -> Any:
        if self.provider_config.type not in {"openai", "openai_compatible"}:
            raise NotImplementedError(
                "OpenAI Agents synthesis backend only supports openai/openai_compatible providers"
            )
        client = sdk.AsyncOpenAI(
            api_key=self._resolve_api_key(),
            base_url=self.provider_config.base_url,
            timeout=float(self.provider_config.timeout_s),
        )
        return sdk.OpenAIChatCompletionsModel(
            model=self.model_ref.model,
            openai_client=client,
        )

    def bind_atomic_tools(
        self,
        *,
        tool_definitions: list[dict[str, Any]],
        tool_executors: dict[str, ToolExecutor],
    ) -> None:
        self.tool_definitions = [dict(definition) for definition in tool_definitions]
        self.tool_executors = dict(tool_executors)

    def _normalized_tool_definitions(self) -> list[dict[str, Any]]:
        return [_normalize_tool_definition(definition) for definition in self.tool_definitions]

    def _build_tools(self) -> list[object]:
        sdk_tools: list[object] = []
        for definition in self._normalized_tool_definitions():
            tool_name = str(definition["name"])
            executor = self.tool_executors.get(tool_name)
            if executor is None:
                continue
            sdk_tools.append(_make_sdk_tool(definition, executor))
        return sdk_tools

    def _build_session(self, sdk: SimpleNamespace, request: SynthesisStageRequest) -> Any | None:
        if not self.runtime_config.sdk_sessions_enabled or self.session_db_path is None:
            return None
        self.session_db_path.parent.mkdir(parents=True, exist_ok=True)
        return sdk.SQLiteSession(
            session_id=f"{request.db_id}:{request.phase.value}:{self.provider_name}",
            db_path=str(self.session_db_path),
        )

    def _write_artifact(
        self,
        *,
        kind: str,
        request: SynthesisStageRequest,
        payload: dict[str, Any],
    ) -> str:
        if self.traces_dir is None:
            return _trace_stub(kind, request, self.provider_name, self.model_name)
        target_dir = self.traces_dir / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / (
            f"{request.db_id}__{request.phase.value}__{self.provider_name}__{self.model_name}.json"
        )
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target_path)

    async def run_stage(self, request: SynthesisStageRequest) -> SynthesisStageResult:
        sdk = _load_sdk_components()
        if hasattr(sdk, "set_tracing_disabled"):
            sdk.set_tracing_disabled(
                disabled=(not self.runtime_config.tracing) or self.provider_config.type != "openai"
            )
        model = self._build_model(sdk)
        tools = self._build_tools() if request.phase == SynthesisPhase.SCHEMA_EXPLORATION else []
        agent = _build_agent(
            sdk,
            request=request,
            model=model,
            structured_output=True,
            tools=tools,
        )
        session = self._build_session(sdk, request)

        request_input = build_synthesis_phase_input(request)
        max_turns = _phase_max_turns(
            request,
            runtime_max_turns=self.runtime_config.max_turns,
            tools=tools,
        )
        started_at = perf_counter()
        try:
            run_result = await sdk.Runner.run(
                agent,
                request_input,
                max_turns=max_turns,
                session=session,
            )
        except Exception as exc:
            if type(exc).__name__ != "ModelBehaviorError":
                raise
            # Some openai-compatible endpoints reject SDK-managed structured output.
            # Retry once with plain text JSON and normalize locally.
            fallback_agent = _build_agent(
                sdk,
                request=request,
                model=model,
                structured_output=False,
                tools=tools,
            )
            run_result = await sdk.Runner.run(
                fallback_agent,
                request_input,
                max_turns=max_turns,
                session=None,
            )
        latency_ms = int((perf_counter() - started_at) * 1000)

        tool_traces = _extract_tool_traces(
            run_result,
            request=request,
            provider=self.provider_name,
            model=self.model_name,
        )
        try:
            payload_model, payload_repair_codes = _normalize_phase_payload(
                request, run_result.final_output
            )
        except Exception as exc:
            self._write_artifact(
                kind="normalize_failures",
                request=request,
                payload={
                    "phase": request.phase.value,
                    "input": request_input,
                    "raw_final_output": run_result.final_output,
                    "latency_ms": latency_ms,
                    "turn_count": _extract_turn_count(run_result),
                    "token_usage": _extract_token_usage(run_result),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
        transcript_ref = self._write_artifact(
            kind="transcripts",
            request=request,
            payload={
                "phase": request.phase.value,
                "input": request_input,
                "final_output": payload_model.model_dump(mode="json"),
                "payload_repair_codes": payload_repair_codes,
                "latency_ms": latency_ms,
                "turn_count": _extract_turn_count(run_result),
                "token_usage": _extract_token_usage(run_result),
            },
        )
        tool_trace_ref = self._write_artifact(
            kind="tool_traces",
            request=request,
            payload={
                "phase": request.phase.value,
                "tool_calls": [trace.model_dump(mode="json") for trace in tool_traces],
                "run_items": [repr(item) for item in getattr(run_result, "new_items", []) or []],
            },
        )
        summary = getattr(payload_model, "memory_summary", f"{request.phase.value} completed")
        memory_entry = SynthesisMemoryEntry(
            phase=request.phase,
            provider=self.provider_name,
            model=self.model_name,
            summary=summary if isinstance(summary, str) else f"{request.phase.value} completed",
            turn_count=_extract_turn_count(run_result),
            token_usage=_extract_token_usage(run_result),
            transcript_ref=transcript_ref,
            tool_trace_ref=tool_trace_ref,
        )
        return SynthesisStageResult(
            phase=request.phase,
            provider=self.provider_name,
            model=self.model_name,
            payload=payload_model,
            payload_repair_codes=payload_repair_codes,
            memory_entry=memory_entry,
            tool_traces=tool_traces,
        )
