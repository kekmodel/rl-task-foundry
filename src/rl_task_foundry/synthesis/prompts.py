"""Prompt builders for the synthesis-agent runtime skeleton."""

from __future__ import annotations

import json

from rl_task_foundry.synthesis.runtime import SynthesisPhase, SynthesisStageRequest


def _phase_atomic_tools_payload(
    request: SynthesisStageRequest,
) -> list[dict[str, object]]:
    if request.phase == SynthesisPhase.ARTIFACT_GENERATION:
        return request.available_atomic_tools
    condensed: list[dict[str, object]] = []
    for tool in request.available_atomic_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        description = tool.get("description", "")
        if not isinstance(name, str):
            continue
        condensed.append(
            {
                "name": name,
                "description": description if isinstance(description, str) else "",
            }
        )
    return condensed


def build_synthesis_phase_instructions(phase: SynthesisPhase) -> str:
    common = (
        "You are one phase of a synthesis meta-agent that creates verifiable database "
        "task environments over a shared per-database atomic tool set. Return only a "
        "single JSON object that matches the required schema. Do not emit markdown or "
        "prose outside the JSON object."
    )
    if phase == SynthesisPhase.SCHEMA_EXPLORATION:
        return (
            f"{common} Analyze the schema summary together with the available atomic "
            "tools, infer domain signals, and identify only the compositional task "
            "categories that can be solved by chaining those tools. Never assume that "
            "new tools can be created."
        )
    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        return (
            f"{common} Choose or validate the best task category for the requested "
            "environment draft, but only if the task remains feasible with the shared "
            "atomic tool set. Base the reasoning on tool composition and unique-answer "
            "task design, not on hypothetical tool additions."
        )
    return (
        f"{common} Produce an artifact-generation payload that contains a valid "
        "`proposed_environment` object with only agent-authored fields and an "
        "`artifacts` object. The environment must rely on the shared atomic tool set "
        "from the input. Do not generate `tool.py` or `tool_self_test.py`. "
        "`solution.py`, `verifier.py`, and `shadow_verifier.py` may call only names "
        "present in `available_atomic_tools`, and the task must have a unique canonical "
        "JSON answer with any needed tie-breakers stated in the question. Return the "
        "current contract shape exactly: `proposed_environment.task`, "
        "`proposed_environment.solution`, `proposed_environment.verifier`, "
        "`proposed_environment.shadow_verifier`, `proposed_environment.instance_space`, "
        "plus `artifacts.solution_source`, `artifacts.verifier_source`, and "
        "`artifacts.shadow_verifier_source`. Do not emit legacy keys such as "
        "`answer_schema`, `canonical_answer_notes`, `difficulty`, `solution.py`, "
        "`verifier.py`, or `shadow_verifier.py`. Replace every example leaf value "
        "with concrete task-specific content; do not copy contract placeholders or "
        "sample SQL verbatim."
    )


def build_synthesis_phase_input(request: SynthesisStageRequest) -> str:
    available_atomic_tools = _phase_atomic_tools_payload(request)
    available_atomic_tool_names = [
        str(tool["name"])
        for tool in available_atomic_tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ]
    payload = {
        "phase": request.phase.value,
        "db_id": request.db_id,
        "atomic_tool_set_ref": request.atomic_tool_set_ref,
        "available_atomic_tools": available_atomic_tools,
        "available_atomic_tool_names": available_atomic_tool_names,
        "available_atomic_tools_role": (
            "database-level shared atomic tool definitions. These are the only tools that "
            "generated solution/verifier/shadow code may reference. Do not invent new "
            "tools or generate tool.py."
        ),
        "domain": {
            "name": request.domain_name,
            "user_role": request.user_role,
            "agent_role": request.agent_role,
            "scenario_description": request.scenario_description,
        },
        "requested_category": (
            request.requested_category.value if request.requested_category is not None else None
        ),
        "attempt_index": request.attempt_index,
        "schema_summary": request.schema_summary,
        "previous_outputs": {
            phase.value: output.model_dump(mode="json")
            for phase, output in request.previous_outputs.items()
        },
        "previous_outputs_role": "authoritative structured outputs from earlier phases",
        "memory": [entry.model_dump(mode="json") for entry in request.memory],
        "memory_role": "compressed execution summaries from earlier phase runs",
        "latest_registration_diagnostics": (
            request.latest_registration_diagnostics.model_dump(mode="json")
            if request.latest_registration_diagnostics is not None
            else None
        ),
        "latest_registration_diagnostics_role": (
            "structured registration feedback from the most recent failed artifact attempt"
        ),
        "latest_self_consistency_diagnostics": (
            request.latest_self_consistency_diagnostics.model_dump(mode="json")
            if request.latest_self_consistency_diagnostics is not None
            else None
        ),
        "latest_self_consistency_diagnostics_role": (
            "structured solution/verifier feedback from the most recent failed artifact attempt"
        ),
        "strongest_difficulty_vector": {
            axis.value: value for axis, value in request.strongest_difficulty_vector.items()
        },
        "difficulty_crank_index": request.difficulty_crank_index,
        "difficulty_crank_history": [axis.value for axis in request.difficulty_crank_history],
        "difficulty_crank_role": (
            "difficulty escalation policy context. Increase at most one difficulty axis per "
            "attempt and never weaken previously established difficulty."
        ),
    }
    if request.phase == SynthesisPhase.ARTIFACT_GENERATION:
        payload["artifact_generation_output_contract"] = {
            "proposed_environment": {
                "task": {
                    "question": (
                        "Which store should serve the selected customer? Return a JSON "
                        "object with the required keys."
                    ),
                    "category": (
                        request.requested_category.value
                        if request.requested_category is not None
                        else "category_taxonomy_value"
                    ),
                    "output_schema": {
                        "version": "v2",
                        "root": {
                            "name": "answer",
                            "type": "object",
                            "fields": [
                                {
                                    "name": "store_id",
                                    "type": "int",
                                    "description": "selected store identifier",
                                }
                            ],
                        },
                        "primary_output_format": "json_object",
                    },
                    "constraint_summary": [
                        {
                            "key": "tie_break",
                            "kind": "uniqueness",
                            "summary": (
                                "If multiple rows satisfy the task, the question must "
                                "state an explicit deterministic tie-break."
                            ),
                        }
                    ],
                    "difficulty_vector": {
                        "slot_count": 1.0,
                    },
                    "instance_parameters": {},
                },
                "solution": {
                    "entrypoint": "solve",
                },
                "verifier": {
                    "entrypoint": "verify",
                    "fetch_facts_function": "fetch_facts",
                    "facts_match_function": "facts_match_answer_claims",
                    "check_constraints_function": "check_constraints",
                    "facts_schema": {"facts": []},
                    "official_judgment": True,
                },
                "shadow_verifier": {
                    "entrypoint": "verify",
                    "fetch_facts_function": "fetch_facts",
                    "facts_match_function": "facts_match_answer_claims",
                    "check_constraints_function": "check_constraints",
                    "facts_schema": {"facts": []},
                    "official_judgment": False,
                },
                "instance_space": {
                    "anchor_query": {
                        "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                        "outputs": ["customer_id"],
                    },
                    "parameters": {},
                    "sampling": {
                        "strategy": "deterministic_hash",
                        "seed": 0,
                    },
                    "instance_count": 1,
                },
            },
            "artifacts": {
                "solution_source": "python source string",
                "verifier_source": "python source string",
                "shadow_verifier_source": "python source string",
            },
        }
        payload["artifact_generation_contract_role"] = (
            "Concrete example of the required JSON shape. Replace every leaf value with "
            "task-specific content and concrete schema fields; never copy the sample "
            "question, sample SQL, or sample field names verbatim unless they are truly "
            "correct for the generated environment."
        )
        payload["artifact_generation_legacy_keys_forbidden"] = [
            "answer_schema",
            "canonical_answer_notes",
            "difficulty",
            "solution.py",
            "verifier.py",
            "shadow_verifier.py",
        ]
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
