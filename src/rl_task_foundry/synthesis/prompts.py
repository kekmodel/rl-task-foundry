"""LLM-friendly prompt builders for the label-first synthesis pipeline."""

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
    "{language}. All schema field names, code identifiers, and tool names must remain in English."
)

ERROR_TEMPLATES = {
    "canonical_answer_schema_mismatch": (
        "The canonical answer could not be validated against the inferred schema. Return a valid "
        "canonical answer JSON object or JSON array."
    ),
    "call_count_limit_exceeded": (
        "The previous task was too easy or too shallow. Increase the latent task difficulty by "
        "following the current crank hint."
    ),
}

FEW_SHOT_EXAMPLE = {
    "label": {
        "canonical_answer_json": '[{"city":"Seoul","day":1,"hotel":"Seoul Station Stay","total_cost":180},{"city":"Suwon","day":2,"hotel":"Suwon Fortress Hotel","total_cost":160}]',
        "anchor_entity": {"anchor_id": 1},
        "difficulty_vector": {
            "search_cost": 2.0,
            "solution_space": 2.0,
            "constraint_density": 2.0,
        },
        "instance_parameters": {},
        "label_summary": "The cheapest valid two-day itinerary is unique after applying connectivity and tie-break rules.",
    },
    "task": {
        "question": "봄 시즌 기준으로 연결된 두 도시를 중복 없이 방문하는 2일 일정을 짜 주세요. 각 날짜마다 가장 저렴한 숙소를 고르고, 하루 총비용이 같으면 도시 이름이 사전순으로 앞서는 쪽을 선택해 주세요.",
        "constraint_summary": [
            {
                "key": "connected_cities_only",
                "kind": "other",
                "summary": "Each next city must be reachable from the previous city.",
            },
            {
                "key": "no_repeat_city",
                "kind": "uniqueness",
                "summary": "Do not repeat a city.",
            },
        ],
        "instance_space": {
            "anchor_query": {
                "sql": "SELECT anchor_id FROM proof_anchors ORDER BY anchor_id",
                "outputs": ["anchor_id"],
            },
            "parameters": {},
            "sampling": {"strategy": "deterministic_hash", "seed": 0},
            "instance_count": 1,
        },
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
            lines.append(f"- {qualified_name}: columns={columns[:8]}")
    return lines


def _phase_relevant_previous_outputs(request: SynthesisStageRequest) -> dict[str, Any]:
    phase_map: dict[SynthesisPhase, list[SynthesisPhase]] = {
        SynthesisPhase.CATEGORY_INFERENCE: [SynthesisPhase.SCHEMA_EXPLORATION],
        SynthesisPhase.LABEL_CONSTRUCTION: [
            SynthesisPhase.SCHEMA_EXPLORATION,
            SynthesisPhase.CATEGORY_INFERENCE,
        ],
        SynthesisPhase.TASK_SYNTHESIS: [SynthesisPhase.LABEL_CONSTRUCTION],
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


def _recent_memory_lines(request: SynthesisStageRequest) -> list[str]:
    entries = request.memory[-3:]
    return [f"- {entry.phase.value}: {entry.summary}" for entry in entries if entry.summary]


def _render_error_feedback(request: SynthesisStageRequest) -> list[str]:
    if request.latest_quality_gate_feedback is None:
        return []
    status = request.latest_quality_gate_feedback.status
    template = ERROR_TEMPLATES.get(status)
    return [template] if template is not None else []


def _render_difficulty_feedback(request: SynthesisStageRequest) -> str | None:
    if request.crank_hint:
        return request.crank_hint
    return None


def _phase_output_contract(phase: SynthesisPhase, request: SynthesisStageRequest) -> dict[str, Any]:
    if phase == SynthesisPhase.SCHEMA_EXPLORATION:
        return {
            "domain_hypothesis": "movie rental support workflows",
            "candidate_topics": ["assignment", "other"],
            "sample_observations": [
                "count_customer() returned 599",
                "top_3_payment_by_amount_desc(limit=3) showed repeated high-value customers",
            ],
            "memory_summary": "grounded schema exploration completed",
        }
    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        return {
            "selected_topic": request.requested_topic or "assignment",
            "rationale": "This topic is feasible with the observed tables, rows, and tool paths.",
            "memory_summary": "topic inference completed",
        }
    if phase == SynthesisPhase.LABEL_CONSTRUCTION:
        return {
            "canonical_answer_json": '{"store_id": 1}',
            "anchor_entity": {"customer_id": 148},
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "instance_parameters": {},
            "label_summary": "The answer is unique after grounded tie-break rules are applied.",
            "memory_summary": "label construction completed",
        }
    return {
        "question": "사용자 입장에서 자연스럽게 묻는 요청문을 작성하세요.",
        "constraint_summary": [
            {
                "key": "stable_tie_break",
                "kind": "uniqueness",
                "summary": "Break ties with a stable deterministic rule.",
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


def build_synthesis_phase_instructions(phase: SynthesisPhase) -> str:
    common = (
        "You are one phase of a synthesis agent that builds grounded RLVR database tasks. "
        "Return exactly one JSON object matching the required output contract. "
        "Do not emit markdown fences or extra commentary."
    )
    if phase == SynthesisPhase.SCHEMA_EXPLORATION:
        return (
            f"{common} This is the data exploration phase. Use the provided function tools to inspect "
            "real database rows and produce concrete sample observations. Do not guess hidden values."
        )
    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        return (
            f"{common} Choose one grounded topic string that matches the observed schema and sample rows."
        )
    if phase == SynthesisPhase.LABEL_CONSTRUCTION:
        return (
            f"{common} Construct the latent label first. Decide the canonical answer JSON, anchor entity, "
            "and difficulty vector from grounded evidence before writing any user-facing request."
        )
    return (
        f"{common} Reverse-design the natural user request from the already-fixed label. "
        "Do not change the canonical answer or difficulty. Surface the constraints naturally in the request."
    )


def build_synthesis_phase_input(request: SynthesisStageRequest) -> str:
    sections: list[str] = []
    sections.append("# Domain\n" f"{request.domain_name}: {request.scenario_description}")
    if request.requested_topic is not None:
        sections.append(f"# Requested Topic\n{request.requested_topic}")

    schema_lines = _schema_orientation_lines(request)
    if schema_lines:
        sections.append("# Schema Orientation\n" + "\n".join(schema_lines))

    previous_outputs = _phase_relevant_previous_outputs(request)
    if previous_outputs:
        sections.append("# Previous Phase Outputs\n" + _json_block(previous_outputs))

    grounded_tool_evidence = _grounded_tool_evidence_lines(request)
    if grounded_tool_evidence:
        sections.append("# Grounded Evidence\n" + "\n".join(grounded_tool_evidence))

    recent_memory = _recent_memory_lines(request)
    if recent_memory:
        sections.append("# Recent Memory\n" + "\n".join(recent_memory))

    if request.phase == SynthesisPhase.TASK_SYNTHESIS:
        sections.append(
            "# User-Facing Language\n"
            + TASK_LANGUAGE_INSTRUCTION.format(language=_language_name(request))
        )

    error_feedback = _render_error_feedback(request)
    if error_feedback:
        sections.append("# Retry Fixes\n" + "\n".join(f"- {line}" for line in error_feedback))

    difficulty_feedback = _render_difficulty_feedback(request)
    if difficulty_feedback:
        sections.append("# Difficulty Guidance\n" + difficulty_feedback)

    if request.phase == SynthesisPhase.TASK_SYNTHESIS:
        sections.append(
            "# One-Shot Example\n"
            "Use this only as a structural example. Do not copy values verbatim.\n"
            + _json_block(FEW_SHOT_EXAMPLE)
        )

    sections.append(
        "# Required Output Contract\n" + _json_block(_phase_output_contract(request.phase, request))
    )
    return "\n\n".join(sections)
