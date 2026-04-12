"""LLM-friendly prompt builders for the synthesis pipeline."""

from __future__ import annotations

import json
from typing import Any

from rl_task_foundry.synthesis.runtime import SynthesisPhase, SynthesisStageRequest

LANGUAGE_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}

TASK_LANGUAGE_INSTRUCTION = (
    "Generate the user-facing question, constraint descriptions, and tie-breaker wording in "
    "{language}. All code (solution.py, verifier.py, shadow_verifier.py), field names, and "
    "tool calls must remain in English."
)

ERROR_TEMPLATES = {
    "fetch_facts_result_not_object": (
        "In the previous attempt, fetch_facts() returned a non-dict value. Return one dict."
    ),
    "top_level_statement_forbidden": (
        "Do not place statements other than imports and function definitions at module top level."
    ),
    "facts_schema_keys_mismatch": (
        "fetch_facts() must return exactly the expected fact keys: {expected_keys}. "
        "The previous attempt returned: {returned_keys}."
    ),
    "shadow_verifier_not_independent": (
        "shadow_verifier must not be structurally identical to verifier. Use a different "
        "verification path."
    ),
    "execution_error": (
        "A runtime execution error occurred: {detail}. Common causes include inventing "
        "unsupported helpers such as tools.query(), using raw SQL inside generated code, or "
        "calling tool names that do not exist."
    ),
    "import_not_allowlisted": "{module_name} is not allowlisted. Use only allowed imports.",
    "call_count_limit_exceeded": (
        "Tool calls exceeded the budget. Rewrite the code to use a more efficient validation path."
    ),
    "check_constraints_constant_boolean_return_forbidden": (
        "check_constraints() must compute its result from facts instead of returning a constant."
    ),
    "fetch_facts_aggregate_call_forbidden": (
        "Do not aggregate inside fetch_facts(). Collect raw facts there and aggregate later."
    ),
    "canonical_answer_schema_mismatch": (
        "The solution output does not match output_schema. Return JSON that matches the schema."
    ),
    "fetch_facts_tools_call_required": (
        "fetch_facts(answer, tools) must call at least one tool to gather DB-grounded facts."
    ),
    "fetch_facts_answer_reference_required": (
        "fetch_facts(answer, tools) must read answer fields and use them to parameterize tool calls."
    ),
    "facts_match_answer_claims_facts_reference_required": (
        "facts_match_answer_claims(answer, facts) must compare answer claims against facts."
    ),
    "check_constraints_facts_reference_required": (
        "check_constraints(answer, facts) must inspect facts directly."
    ),
    "tool_return_key_not_in_returns_schema": (
        "Only access keys that are declared in the tool returns_schema."
    ),
    "tool_return_key_access_forbidden": (
        "Only access named fields on object-valued tool results whose returns_schema declares them."
    ),
}

FEW_SHOT_TRIP_EXAMPLE = {
    "category": "itinerary",
    "label": {
        "canonical_answer_json": '[{"day":1,"city":"Seoul","hotel":"Station Stay"},{"day":2,"city":"Suwon","hotel":"Fortress Hotel"}]',
        "output_schema": {
            "version": "v2",
            "root": {
                "name": "itinerary",
                "type": "list",
                "ordered": True,
                "items": {
                    "name": "entry",
                    "type": "object",
                    "fields": [
                        {"name": "day", "type": "int"},
                        {"name": "city", "type": "str"},
                        {"name": "hotel", "type": "str"},
                    ],
                },
            },
            "primary_output_format": "json_array",
        },
        "difficulty_vector": {
            "search_cost": 2.0,
            "solution_space": 2.0,
            "constraint_density": 2.0,
        },
        "instance_parameters": {},
    },
    "task": {
        "question": "Plan a 2-day spring itinerary. Visit connected cities without repetition and choose the cheapest valid lodging each day. Break ties by city name.",
        "constraint_summary": [
            {
                "key": "connected_cities_only",
                "kind": "other",
                "summary": "Each next city must be reachable from the previous city through the city-link relation.",
            },
            {
                "key": "no_repeat_city",
                "kind": "uniqueness",
                "summary": "Do not repeat a city.",
            },
        ],
        "instance_space": {
            "anchor_query": {
                "sql": "SELECT city_id FROM city ORDER BY city_id",
                "outputs": ["city_id"],
            },
            "parameters": {},
            "sampling": {"strategy": "deterministic_hash", "seed": 0},
            "instance_count": 1,
        },
    },
    "proposed_environment": {
        "task": {
            "question": "Plan a 2-day spring itinerary. Visit connected cities without repetition and choose the cheapest valid lodging each day. Break ties by city name.",
            "category": "itinerary",
            "output_schema": {
                "version": "v2",
                "root": {
                    "name": "itinerary",
                    "type": "list",
                    "ordered": True,
                    "items": {
                        "name": "entry",
                        "type": "object",
                        "fields": [
                            {"name": "day", "type": "int"},
                            {"name": "city", "type": "str"},
                            {"name": "hotel", "type": "str"},
                        ],
                    },
                },
                "primary_output_format": "json_array",
            },
            "constraint_summary": [
                {
                    "key": "connected_cities_only",
                    "kind": "other",
                    "summary": "Each next city must be reachable from the previous city through the city-link relation.",
                },
                {
                    "key": "no_repeat_city",
                    "kind": "uniqueness",
                    "summary": "Do not repeat a city.",
                },
            ],
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 2.0,
                "constraint_density": 2.0,
            },
            "instance_parameters": {},
        },
        "solution": {"entrypoint": "solve"},
        "verifier": {
            "entrypoint": "verify",
            "fetch_facts_function": "fetch_facts",
            "facts_match_function": "facts_match_answer_claims",
            "check_constraints_function": "check_constraints",
            "facts_schema": {
                "facts": [
                    {
                        "key": "city_path",
                        "entity_ref": "candidate_itinerary",
                        "attribute": "city",
                        "value_type": "str",
                        "nullable": False,
                        "cardinality": "many",
                    },
                    {
                        "key": "hotel_path",
                        "entity_ref": "candidate_itinerary",
                        "attribute": "hotel",
                        "value_type": "str",
                        "nullable": False,
                        "cardinality": "many",
                    },
                ]
            },
        },
        "shadow_verifier": {
            "entrypoint": "verify",
            "fetch_facts_function": "fetch_facts",
            "facts_match_function": "facts_match_answer_claims",
            "check_constraints_function": "check_constraints",
            "facts_schema": {
                "facts": [
                    {
                        "key": "city_path",
                        "entity_ref": "candidate_itinerary",
                        "attribute": "city",
                        "value_type": "str",
                        "nullable": False,
                        "cardinality": "many",
                    },
                    {
                        "key": "hotel_path",
                        "entity_ref": "candidate_itinerary",
                        "attribute": "hotel",
                        "value_type": "str",
                        "nullable": False,
                        "cardinality": "many",
                    },
                ]
            },
        },
        "instance_space": {
            "anchor_query": {
                "sql": "SELECT city_id FROM city ORDER BY city_id",
                "outputs": ["city_id"],
            },
            "parameters": {},
            "sampling": {"strategy": "deterministic_hash", "seed": 0},
            "instance_count": 1,
        },
    },
    "artifacts": {
        "solution_source": "def solve(tools):\n    # grounded chain over city -> lodging\n    return []\n",
        "verifier_source": "def fetch_facts(answer, tools):\n    return {}\n\n\ndef facts_match_answer_claims(answer, facts):\n    return True\n\n\ndef check_constraints(answer, facts):\n    return True\n\n\ndef verify(answer, tools):\n    facts = fetch_facts(answer, tools)\n    return facts_match_answer_claims(answer, facts) and check_constraints(answer, facts)\n",
        "shadow_verifier_source": "def fetch_facts(answer, tools):\n    return {}\n\n\ndef facts_match_answer_claims(answer, facts):\n    return True\n\n\ndef check_constraints(answer, facts):\n    return True\n\n\ndef verify(answer, tools):\n    facts = fetch_facts(answer, tools)\n    return facts_match_answer_claims(answer, facts) and check_constraints(answer, facts)\n",
    },
}


def _json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _language_name(request: SynthesisStageRequest) -> str:
    return LANGUAGE_NAMES.get(request.task_language, request.task_language)


def _schema_orientation_lines(request: SynthesisStageRequest) -> list[str]:
    schema_summary = request.schema_summary or {}
    tables = schema_summary.get("tables")
    lines: list[str] = []
    if isinstance(schema_summary.get("table_count"), int):
        lines.append(f"- Table count: {schema_summary['table_count']}")
    if isinstance(schema_summary.get("edge_count"), int):
        lines.append(f"- Foreign-key edge count: {schema_summary['edge_count']}")
    if isinstance(tables, list):
        for table in tables[:8]:
            if not isinstance(table, dict):
                continue
            qualified_name = table.get("qualified_name") or table.get("table_name")
            columns = table.get("column_names") or []
            pk = table.get("primary_key") or []
            lines.append(
                f"- {qualified_name} | pk={pk} | columns={columns[:8]}"
            )
    return lines


def _phase_relevant_previous_outputs(request: SynthesisStageRequest) -> dict[str, Any]:
    phase_map: dict[SynthesisPhase, list[SynthesisPhase]] = {
        SynthesisPhase.CATEGORY_INFERENCE: [SynthesisPhase.SCHEMA_EXPLORATION],
        SynthesisPhase.LABEL_CONSTRUCTION: [
            SynthesisPhase.SCHEMA_EXPLORATION,
            SynthesisPhase.CATEGORY_INFERENCE,
        ],
        SynthesisPhase.TASK_SYNTHESIS: [SynthesisPhase.LABEL_CONSTRUCTION],
        SynthesisPhase.ARTIFACT_GENERATION: [
            SynthesisPhase.LABEL_CONSTRUCTION,
            SynthesisPhase.TASK_SYNTHESIS,
        ],
    }
    wanted = phase_map.get(request.phase, [])
    return {
        phase.value: request.previous_outputs[phase].model_dump(mode="json")
        for phase in wanted
        if phase in request.previous_outputs
    }


def _grounded_tool_evidence_lines(request: SynthesisStageRequest) -> list[str]:
    schema_output = request.previous_outputs.get(SynthesisPhase.SCHEMA_EXPLORATION)
    if schema_output is None:
        return []
    sample_observations = getattr(schema_output, "sample_observations", [])
    if not isinstance(sample_observations, list):
        return []
    return [f"- {line}" for line in sample_observations[:8] if isinstance(line, str) and line.strip()]


def _leaf_output_field_names(field: dict[str, Any]) -> set[str]:
    field_type = field.get("type")
    if field_type == "object":
        names: set[str] = set()
        for child in field.get("fields", []):
            if isinstance(child, dict):
                names.update(_leaf_output_field_names(child))
        return names
    if field_type == "list":
        items = field.get("items")
        if isinstance(items, dict):
            return _leaf_output_field_names(items)
        return set()
    field_name = field.get("name")
    if isinstance(field_name, str) and field_name:
        return {field_name}
    return set()


def _json_schema_object_keys(schema: Any) -> set[str]:
    if not isinstance(schema, dict):
        return set()
    keys: set[str] = set()
    if isinstance(schema.get("properties"), dict):
        keys.update(str(key) for key in schema["properties"])
    if isinstance(schema.get("items"), dict):
        keys.update(_json_schema_object_keys(schema["items"]))
    for branch in schema.get("anyOf", []):
        keys.update(_json_schema_object_keys(branch))
    return keys


def _relevant_atomic_tool_lines(request: SynthesisStageRequest) -> list[str]:
    if not request.available_atomic_tools:
        return []
    anchor_outputs: set[str] = set()
    output_field_names: set[str] = set()
    observed_tool_names: set[str] = set()

    task_output = request.previous_outputs.get(SynthesisPhase.TASK_SYNTHESIS)
    if task_output is not None:
        instance_space = getattr(task_output, "instance_space", None)
        anchor_query = getattr(instance_space, "anchor_query", None)
        outputs = getattr(anchor_query, "outputs", None)
        if isinstance(outputs, list):
            anchor_outputs = {str(name) for name in outputs}

    label_output = request.previous_outputs.get(SynthesisPhase.LABEL_CONSTRUCTION)
    if label_output is not None:
        output_schema = label_output.output_schema.model_dump(mode="json")
        root = output_schema.get("root")
        if isinstance(root, dict):
            output_field_names = _leaf_output_field_names(root)

    for line in _grounded_tool_evidence_lines(request):
        stripped = line.removeprefix("- ").strip()
        if "(" in stripped:
            observed_tool_names.add(stripped.split("(", 1)[0].strip())

    ranked: list[tuple[int, str]] = []
    for definition in request.available_atomic_tools:
        if not isinstance(definition, dict):
            continue
        name = definition.get("name")
        if not isinstance(name, str) or not name:
            continue
        params_schema = definition.get("params_schema")
        returns_schema = definition.get("returns_schema")
        param_names = (
            set(params_schema.get("properties", {}).keys())
            if isinstance(params_schema, dict)
            else set()
        )
        return_keys = _json_schema_object_keys(returns_schema)
        score = 0
        if anchor_outputs & param_names:
            score += 3
        if output_field_names & return_keys:
            score += 3
        if name in observed_tool_names:
            score += 2
        if any(token in name for token in sorted(anchor_outputs | output_field_names)):
            score += 1
        if score <= 0:
            continue
        params_rendered = ", ".join(sorted(param_names)) if param_names else "no params"
        return_keys_rendered = ", ".join(sorted(return_keys)) if return_keys else "scalar/null"
        ranked.append(
            (
                -score,
                f"- {name}({params_rendered}) -> keys [{return_keys_rendered}]",
            )
        )
    ranked.sort()
    return [line for _, line in ranked[:12]]


def _recent_memory_lines(request: SynthesisStageRequest) -> list[str]:
    entries = request.memory[-3:]
    return [
        f"- {entry.phase.value}: {entry.summary}"
        for entry in entries
        if entry.summary
    ]


def _registration_fact_key_mismatches(
    request: SynthesisStageRequest,
) -> list[dict[str, object]]:
    diagnostics = request.latest_registration_diagnostics
    if diagnostics is None:
        return []

    mismatches: list[dict[str, object]] = []
    for artifact_name in ("verifier", "shadow_verifier"):
        artifact = getattr(diagnostics, artifact_name, None)
        if artifact is None:
            continue
        error_codes = {
            *artifact.static_error_codes,
            *artifact.execution_error_codes,
            *artifact.probe_error_codes,
        }
        if "facts_schema_keys_mismatch" not in error_codes:
            continue
        mismatches.append(
            {
                "artifact": artifact_name,
                "expected_keys": list(artifact.probe_expected_fact_keys),
                "returned_keys": list(artifact.probe_fetch_facts_return_keys),
                "missing_keys": list(artifact.probe_missing_fact_keys),
                "extra_keys": list(artifact.probe_extra_fact_keys),
            }
        )
    return mismatches


def _render_error_feedback(request: SynthesisStageRequest) -> list[str]:
    diagnostics = request.latest_registration_diagnostics
    if diagnostics is None:
        return []

    mismatch_by_artifact = {
        mismatch["artifact"]: mismatch for mismatch in _registration_fact_key_mismatches(request)
    }

    rendered: list[str] = []
    for code in diagnostics.error_codes:
        template = ERROR_TEMPLATES.get(code)
        if template is None:
            rendered.append(f"Resolve this error before the next attempt: {code}.")
            continue
        if code == "facts_schema_keys_mismatch" and mismatch_by_artifact:
            for artifact_name, mismatch in mismatch_by_artifact.items():
                rendered.append(
                    f"{artifact_name}: "
                    + template.format(
                        expected_keys=", ".join(mismatch["expected_keys"]),
                        returned_keys=", ".join(mismatch["returned_keys"]),
                    )
                )
            continue
        rendered.append(
            template.format(
                expected_keys="",
                returned_keys="",
                detail="see registration diagnostics",
                module_name="forbidden module",
            )
        )
    return rendered


def _render_difficulty_feedback(request: SynthesisStageRequest) -> str | None:
    if request.latest_quality_gate_feedback is None and request.crank_hint is None:
        return None
    lines: list[str] = []
    if request.latest_quality_gate_feedback is not None:
        feedback = request.latest_quality_gate_feedback
        lines.append(
            "The previous synthesized task was outside the solver pass-rate band. "
            f"Status: {feedback.status}. Pass rate: {feedback.pass_rate:.2f} "
            f"(CI {feedback.ci_lower:.2f}-{feedback.ci_upper:.2f})."
        )
        if feedback.previous_question:
            lines.append(f"Do not repeat this previous question: {feedback.previous_question}")
        if feedback.previous_semantic_dedup_text:
            lines.append(
                "Do not repeat this previous latent task signature: "
                f"{feedback.previous_semantic_dedup_text}"
            )
    if request.crank_hint:
        lines.append(request.crank_hint)
    return "\n".join(lines) if lines else None


def _fact_value_type_for_output_type(type_name: str | None) -> str:
    mapping = {
        "string": "str",
        "str": "str",
        "enum": "str",
        "int": "int",
        "integer": "int",
        "float": "float",
        "number": "float",
        "bool": "bool",
        "boolean": "bool",
        "date": "date",
        "datetime": "datetime",
    }
    return mapping.get(type_name or "", "str")


def _collect_fact_examples_from_output_field(
    field: dict[str, Any],
    *,
    cardinality: str = "one",
    entity_ref: str = "answer",
) -> list[dict[str, Any]]:
    field_type = field.get("type")
    if field_type == "object":
        facts: list[dict[str, Any]] = []
        for child in field.get("fields", []):
            if isinstance(child, dict):
                facts.extend(
                    _collect_fact_examples_from_output_field(
                        child,
                        cardinality=cardinality,
                        entity_ref=entity_ref,
                    )
                )
        return facts
    if field_type == "list":
        items = field.get("items")
        if isinstance(items, dict):
            return _collect_fact_examples_from_output_field(
                items,
                cardinality="many",
                entity_ref="answer_item",
            )
        return []
    field_name = field.get("name")
    if not isinstance(field_name, str) or not field_name:
        return []
    return [
        {
            "key": field_name,
            "entity_ref": entity_ref,
            "attribute": field_name,
            "value_type": _fact_value_type_for_output_type(field_type),
            "nullable": bool(field.get("nullable", False)),
            "cardinality": cardinality,
        }
    ]


def _artifact_fact_schema_example(request: SynthesisStageRequest) -> dict[str, Any]:
    label_output = request.previous_outputs.get(SynthesisPhase.LABEL_CONSTRUCTION)
    if isinstance(label_output, object) and hasattr(label_output, "output_schema"):
        output_schema = label_output.output_schema.model_dump(mode="json")
        root = output_schema.get("root")
        if isinstance(root, dict):
            facts = _collect_fact_examples_from_output_field(root)
            if facts:
                return {"facts": facts[:6]}
    return {
        "facts": [
            {
                "key": "store_id",
                "entity_ref": "answer",
                "attribute": "store_id",
                "value_type": "int",
                "nullable": False,
                "cardinality": "one",
            }
        ]
    }


def _phase_output_contract(phase: SynthesisPhase, request: SynthesisStageRequest) -> dict[str, Any]:
    if phase == SynthesisPhase.SCHEMA_EXPLORATION:
        return {
            "domain_hypothesis": "movie rental support workflows",
            "candidate_categories": ["assignment", "other"],
            "sample_observations": [
                "count_customer() returned 599",
                "top_k_payment_by_amount_desc(limit=3) revealed customer and amount patterns",
            ],
            "memory_summary": "grounded schema and data exploration completed",
        }
    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        return {
            "selected_category": (
                request.requested_category.value
                if request.requested_category is not None
                else "assignment"
            ),
            "rationale": "This category remains feasible with the grounded tool composition.",
            "memory_summary": "category inference completed",
        }
    if phase == SynthesisPhase.LABEL_CONSTRUCTION:
        return {
            "canonical_answer_json": '{"store_id": 1}',
            "output_schema": {
                "version": "v2",
                "root": {
                    "name": "answer",
                    "type": "object",
                    "fields": [{"name": "store_id", "type": "int"}],
                },
                "primary_output_format": "json_object",
            },
            "difficulty_vector": {
                "search_cost": 1.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "instance_parameters": {},
            "label_summary": "The label is grounded in concrete observed rows and has a unique answer.",
            "memory_summary": "label construction completed",
        }
    if phase == SynthesisPhase.TASK_SYNTHESIS:
        return {
            "question": "Return the store_id implied by the grounded customer-support scenario.",
            "constraint_summary": [
                {
                    "key": "tie_break",
                    "kind": "uniqueness",
                    "summary": "If multiple candidates remain, choose the smallest store_id.",
                }
            ],
            "instance_space": {
                "anchor_query": {
                    "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                    "outputs": ["customer_id"],
                },
                "parameters": {},
                "sampling": {"strategy": "deterministic_hash", "seed": 0},
                "instance_count": 1,
            },
            "memory_summary": "task synthesis completed",
        }
    label_output = request.previous_outputs.get(SynthesisPhase.LABEL_CONSTRUCTION)
    task_output = request.previous_outputs.get(SynthesisPhase.TASK_SYNTHESIS)
    output_schema: Any = {
        "version": "v2",
        "root": {
            "name": "answer",
            "type": "object",
            "fields": [{"name": "store_id", "type": "int"}],
        },
        "primary_output_format": "json_object",
    }
    difficulty_vector: Any = {
        "search_cost": 1.0,
        "solution_space": 1.0,
        "constraint_density": 1.0,
    }
    instance_parameters: Any = {}
    question = "Return the correct JSON answer for the grounded scenario."
    constraint_summary: Any = [
        {
            "key": "tie_break",
            "kind": "uniqueness",
            "summary": "If multiple candidates remain, choose the smallest stable key.",
        }
    ]
    instance_space: Any = {
        "anchor_query": {
            "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
            "outputs": ["customer_id"],
        },
        "parameters": {},
        "sampling": {"strategy": "deterministic_hash", "seed": 0},
        "instance_count": 1,
    }
    if label_output is not None:
        output_schema = label_output.output_schema.model_dump(mode="json")
        difficulty_vector = label_output.difficulty_vector.model_dump(mode="json")
        instance_parameters = label_output.instance_parameters
    if task_output is not None:
        question = task_output.question
        constraint_summary = [
            item.model_dump(mode="json") for item in task_output.constraint_summary
        ]
        instance_space = task_output.instance_space.model_dump(mode="json")
    fact_schema = _artifact_fact_schema_example(request)
    return {
        "proposed_environment": {
            "task": {
                "question": question,
                "category": (
                    request.requested_category.value
                    if request.requested_category is not None
                    else "assignment"
                ),
                "output_schema": output_schema,
                "constraint_summary": constraint_summary,
                "difficulty_vector": difficulty_vector,
                "instance_parameters": instance_parameters,
            },
            "solution": {"entrypoint": "solve"},
            "verifier": {
                "entrypoint": "verify",
                "fetch_facts_function": "fetch_facts",
                "facts_match_function": "facts_match_answer_claims",
                "check_constraints_function": "check_constraints",
                "facts_schema": fact_schema,
            },
            "shadow_verifier": {
                "entrypoint": "verify",
                "fetch_facts_function": "fetch_facts",
                "facts_match_function": "facts_match_answer_claims",
                "check_constraints_function": "check_constraints",
                "facts_schema": fact_schema,
            },
            "instance_space": instance_space,
        },
        "artifacts": {
            "solution_source": "python source string",
            "verifier_source": "python source string",
            "shadow_verifier_source": "python source string",
        },
    }


def build_synthesis_phase_instructions(phase: SynthesisPhase) -> str:
    common = (
        "You are one phase of a synthesis agent that creates verifiable database task "
        "environments for RLVR training. Always return exactly one JSON object that matches "
        "the required output contract. Do not emit markdown fences or extra prose."
    )
    if phase == SynthesisPhase.SCHEMA_EXPLORATION:
        return (
            f"{common} This is the grounded data-exploration phase. Use the provided tools to "
            "inspect real database rows or aggregates before deciding anything. You must ground "
            "the result in actual tool calls and produce non-empty "
            "`sample_observations`. Use a small number of high-signal tool calls and stop as "
            "soon as you can justify one grounded category proposal and one answerable data "
            "path. Do not assume new tools can be created."
        )
    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        return (
            f"{common} Choose the best feasible category from the grounded exploration output. "
            "Do not invent unsupported categories or hypothetical tools."
        )
    if phase == SynthesisPhase.LABEL_CONSTRUCTION:
        return (
            f"{common} Construct the latent label first. Decide the canonical answer JSON, "
            "output schema, and difficulty vector from grounded evidence before any user-facing "
            "question or code is written."
        )
    if phase == SynthesisPhase.TASK_SYNTHESIS:
        return (
            f"{common} Reverse-design the user-facing task from the already-fixed label. "
            "Do not change the label, schema, or difficulty; only surface them naturally."
        )
    return (
        f"{common} This is the code-generation phase. The label and task outputs from prior "
        "phases are authoritative. Keep generated code aligned with them, use only English for "
        "code and field names. Do not invent new tools. `proposed_environment.task` must fully "
        "restate the authoritative question, output_schema, constraint_summary, difficulty_vector, "
        "and instance_parameters. `verifier.facts_schema` and `shadow_verifier.facts_schema` must "
        "declare the exact keys returned by fetch_facts(). Generated code may call only named "
        "atomic tools through the `tools` object, for example `tools.get_customer_by_id(...)` or "
        "`await tools.get_customer_by_id(...)`. Never use raw SQL, `tools.query(...)`, "
        "`tools.execute(...)`, or invented helper APIs."
    )


def build_synthesis_phase_input(request: SynthesisStageRequest) -> str:
    sections: list[str] = []

    sections.append("# Domain\n" f"{request.domain_name}: {request.scenario_description}")

    if request.requested_category is not None:
        sections.append(f"# Requested Category\n{request.requested_category.value}")

    schema_lines = _schema_orientation_lines(request)
    if schema_lines:
        sections.append("# Schema Orientation\n" + "\n".join(schema_lines))

    if request.phase in {
        SynthesisPhase.TASK_SYNTHESIS,
        SynthesisPhase.ARTIFACT_GENERATION,
    }:
        sections.append(
            "# User-Facing Language\n"
            + TASK_LANGUAGE_INSTRUCTION.format(language=_language_name(request))
        )

    previous_outputs = _phase_relevant_previous_outputs(request)
    if previous_outputs:
        sections.append("# Previous Phase Outputs\n" + _json_block(previous_outputs))

    if request.phase == SynthesisPhase.ARTIFACT_GENERATION:
        label_output = request.previous_outputs.get(SynthesisPhase.LABEL_CONSTRUCTION)
        task_output = request.previous_outputs.get(SynthesisPhase.TASK_SYNTHESIS)
        if label_output is not None and task_output is not None:
            sections.append(
                "# Authoritative Task Contract\n"
                + _json_block(
                    {
                        "task": {
                            "question": task_output.question,
                            "category": (
                                request.requested_category.value
                                if request.requested_category is not None
                                else "assignment"
                            ),
                            "output_schema": label_output.output_schema.model_dump(mode="json"),
                            "constraint_summary": [
                                item.model_dump(mode="json")
                                for item in task_output.constraint_summary
                            ],
                            "difficulty_vector": label_output.difficulty_vector.model_dump(
                                mode="json"
                            ),
                            "instance_parameters": label_output.instance_parameters,
                        },
                        "instance_space": task_output.instance_space.model_dump(mode="json"),
                    }
                )
            )
        grounded_tool_evidence = _grounded_tool_evidence_lines(request)
        if grounded_tool_evidence:
            sections.append("# Grounded Tool Evidence\n" + "\n".join(grounded_tool_evidence))
        relevant_tool_lines = _relevant_atomic_tool_lines(request)
        if relevant_tool_lines:
            sections.append("# Relevant Atomic Tools\n" + "\n".join(relevant_tool_lines))

    recent_memory = _recent_memory_lines(request)
    if recent_memory:
        sections.append("# Recent Memory\n" + "\n".join(recent_memory))

    if request.phase == SynthesisPhase.SCHEMA_EXPLORATION:
        sections.append(
            "# Exploration Goal\n"
            "Inspect the real database through the provided SDK tools. Ground your answer in "
            "actual observed rows or aggregates. Do not guess hidden values."
        )
    elif request.phase == SynthesisPhase.CATEGORY_INFERENCE:
        sections.append(
            "# Category Goal\n"
            "Choose the best feasible category using grounded exploration results."
        )
    elif request.phase == SynthesisPhase.LABEL_CONSTRUCTION:
        sections.append(
            "# Label Goal\n"
            "Decide the canonical answer first. The answer must be grounded in observed data and "
            "must be directly canonicalizable against the output schema."
        )
    elif request.phase == SynthesisPhase.TASK_SYNTHESIS:
        sections.append(
            "# Task Goal\n"
            "Write a natural user-facing question and constraint summaries that surface the fixed "
            "label semantics without changing them."
        )
    else:
        sections.append(
            "# Code Goal\n"
            "Generate solution.py, verifier.py, and shadow_verifier.py that reproduce and verify "
            "the already-fixed label. Do not rely on text reasoning for correctness. The code must "
            "implement the authoritative task contract exactly and each verifier must return facts "
            "matching its declared facts_schema. Use only named atomic tools through the `tools` "
            "object. Never use raw SQL or invented helpers such as tools.query(...)."
        )

    error_feedback = _render_error_feedback(request)
    if error_feedback:
        sections.append("# Retry Fixes\n" + "\n".join(f"- {line}" for line in error_feedback))

    difficulty_feedback = _render_difficulty_feedback(request)
    if difficulty_feedback:
        sections.append("# Difficulty Guidance\n" + difficulty_feedback)

    if request.phase in {SynthesisPhase.TASK_SYNTHESIS, SynthesisPhase.ARTIFACT_GENERATION}:
        sections.append(
            "# One-Shot Example\n"
            "Use this only as a structural example. Do not copy values verbatim.\n"
            + _json_block(FEW_SHOT_TRIP_EXAMPLE)
        )

    sections.append(
        "# Required Output Contract\n" + _json_block(_phase_output_contract(request.phase, request))
    )
    return "\n\n".join(sections)
