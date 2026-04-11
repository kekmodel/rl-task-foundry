"""Prompt builders for the synthesis-agent runtime skeleton."""

from __future__ import annotations

import json

from rl_task_foundry.synthesis.runtime import SynthesisPhase, SynthesisStageRequest


def build_synthesis_phase_instructions(phase: SynthesisPhase) -> str:
    common = (
        "You are one phase of a synthesis meta-agent that creates verifiable database "
        "task environments. Return only a JSON object that matches the required schema."
    )
    if phase == SynthesisPhase.SCHEMA_EXPLORATION:
        return (
            f"{common} Analyze the schema summary, infer domain signals, and identify "
            "which compositional task categories the database can plausibly support."
        )
    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        return (
            f"{common} Choose or validate the best task category for the requested "
            "environment draft and explain the reasoning in structured JSON."
        )
    return (
        f"{common} Produce an artifact-generation payload that contains a valid "
        "`proposed_environment` object with only agent-authored fields and an "
        "`artifacts` object."
    )


def build_synthesis_phase_input(request: SynthesisStageRequest) -> str:
    payload = {
        "phase": request.phase.value,
        "db_id": request.db_id,
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
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
