"""Model-generated task question synthesis and coherence validation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from rl_task_foundry.config.models import DomainConfig, ModelRef, ProviderConfig
from rl_task_foundry.infra.json_chat_client import (
    JsonChatCompletionError,
    request_json_chat_completion,
)
from rl_task_foundry.schema.path_catalog import PathSpec
from rl_task_foundry.tasks.models import TaskSpec


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuestionGenerationError(RuntimeError):
    """Raised when question synthesis cannot be validated."""


class QuestionResponse(StrictModel):
    question: str


class QuestionCoherenceResponse(StrictModel):
    pass_coherence: bool
    reason: str = ""


def _family_shape_guidance(
    *,
    task: TaskSpec,
    question_context: dict[str, object],
) -> dict[str, object]:
    style_guide = question_context.get("question_style_guide")
    if not isinstance(style_guide, dict):
        style_guide = {}
    family_patterns = style_guide.get("family_patterns")
    if not isinstance(family_patterns, dict):
        family_patterns = {}
    family_hint = family_patterns.get(task.question_family)
    if not isinstance(family_hint, dict):
        family_hint = {}
    return {
        "preferred_subject_reference": style_guide.get("preferred_subject_reference"),
        "anchor_reference": style_guide.get("anchor_reference"),
        "downstream_reference": style_guide.get("downstream_reference"),
        "target_reference": style_guide.get("target_reference"),
        "answer_concept_reference": style_guide.get("answer_concept_reference"),
        "count_target_reference": style_guide.get("count_target_reference"),
        "count_phrase_reference": style_guide.get("count_phrase_reference"),
        "count_unit_hint": style_guide.get("count_unit_hint"),
        "answer_labels": style_guide.get("answer_labels", []),
        "answer_shape": style_guide.get("answer_shape"),
        "family_goal": family_hint.get("goal"),
        "preferred_shape": family_hint.get("preferred_shape"),
        "avoid_shape": family_hint.get("avoid_shape"),
        "banned_phrases": style_guide.get("banned_phrases", []),
    }


def _quality_examples(
    *,
    domain: DomainConfig,
    task: TaskSpec,
    question_context: dict[str, object],
) -> dict[str, list[str]]:
    answer_shape = str(question_context.get("answer_shape", "scalar"))
    style_guide = question_context.get("question_style_guide")
    if not isinstance(style_guide, dict):
        style_guide = {}
    count_target = str(style_guide.get("count_target_reference", "item")).strip() or "item"
    count_phrase = str(style_guide.get("count_phrase_reference", "")).strip()
    count_unit = str(style_guide.get("count_unit_hint", "items")).strip() or "items"
    if domain.language == "ko":
        common_avoid = [
            "연결된 값을 확인해줘.",
            "관련된 항목이 몇 개인지 확인해줘.",
            "경로를 따라 값을 알려줘.",
            "제 고객 정보에 연결된 값을 알려주세요.",
        ]
        if task.question_family == "status_lookup":
            return {
                "avoid": common_avoid,
                "prefer": [
                    "현재 등록된 정보를 확인해 주실 수 있나요?",
                    "지금 상태가 어떻게 되어 있는지 알려주세요.",
                ],
            }
        if task.question_family == "causal_chain":
            return {
                "avoid": common_avoid,
                "prefer": [
                    "제 경우에 실제로 어떤 옵션들이 해당되는지 알려주세요.",
                    "이 상황에서 적용되는 대상이 어떤 것들인지 알고 싶어요.",
                ]
                if answer_shape == "list"
                else [
                    "제 경우에 실제로 어떤 항목이 적용되는지 알려주세요.",
                    "이 상황에서 최종적으로 어떤 결과가 되는지 알고 싶어요.",
                ],
            }
        if task.question_family == "aggregate_verification":
            if count_unit == "people":
                return {
                    "avoid": common_avoid,
                    "prefer": [
                        "이 건을 처리하는 사람이 몇 명인지 알려주세요.",
                        "제 상황과 관련된 인원이 몇 명인지 확인해 주세요.",
                    ],
                }
            if count_unit == "places":
                return {
                    "avoid": common_avoid,
                    "prefer": [
                        "이와 관련된 장소가 몇 곳인지 알려주세요.",
                        "제 상황에 연결된 위치가 몇 곳인지 확인해 주세요.",
                    ],
                }
            if count_unit == "cases":
                return {
                    "avoid": common_avoid,
                    "prefer": [
                        "이와 관련된 건이 몇 건인지 알려주세요.",
                        "제 상황에 해당하는 건수가 얼마나 되는지 확인해 주세요.",
                    ],
                }
            return {
                "avoid": common_avoid,
                "prefer": [
                    f"관련된 {count_phrase or count_target}이 얼마나 되는지 알려주세요.",
                    f"제가 확인할 수 있는 {count_target} 수를 알려주세요.",
                ],
            }
        return {
            "avoid": common_avoid,
            "prefer": [
                "최근 시점이 언제인지 알려주세요.",
                "가장 최근에 반영된 시점을 확인해 주세요.",
            ],
        }

    common_avoid = [
        "Check the connected value.",
        "Tell me the related item count.",
        "Follow the path and report the final value.",
        "Tell me the value linked to my customer info.",
    ]
    if task.question_family == "status_lookup":
        return {
            "avoid": common_avoid,
            "prefer": [
                "Could you confirm the currently registered information for me?",
                "Can you tell me what the current status is?",
            ],
        }
    if task.question_family == "causal_chain":
        return {
            "avoid": common_avoid,
            "prefer": [
                "Can you tell me which options actually apply in my case?",
                "I want to know which concrete items I end up with here.",
            ]
            if answer_shape == "list"
            else [
                "Can you tell me which concrete detail actually applies in my case?",
                "I want to know what this ultimately turns into for me.",
            ],
        }
    if task.question_family == "aggregate_verification":
        if count_unit == "people":
            return {
                "avoid": common_avoid,
                "prefer": [
                    "How many people are involved in this for me?",
                    "Can you tell me how many people are handling this?",
                ],
            }
        if count_unit == "places":
            return {
                "avoid": common_avoid,
                "prefer": [
                    "How many locations are involved in this for me?",
                    "Can you tell me how many places are tied to this?",
                ],
            }
        if count_unit == "cases":
            return {
                "avoid": common_avoid,
                "prefer": [
                    "How many relevant cases are associated with this?",
                    "Can you tell me how many relevant cases there are?",
                ],
            }
        return {
            "avoid": common_avoid,
            "prefer": [
                f"How many {count_phrase or count_target} are associated with this for me?",
                f"Can you tell me how many relevant {count_target} there are?",
            ],
        }
    return {
        "avoid": common_avoid,
        "prefer": [
            "When did this most recently happen?",
            "What is the latest time this was updated?",
        ],
    }


def _question_prompt(
    *,
    domain: DomainConfig,
    task: TaskSpec,
    question_context: dict[str, object],
    feedback: str | None = None,
) -> list[dict[str, str]]:
    style_guide = _family_shape_guidance(task=task, question_context=question_context)
    language_instruction = (
        "Write the question in natural Korean as a realistic end-user request."
        if domain.language == "ko"
        else "Write the question in natural English as a realistic end-user request."
    )
    system_prompt = (
        f"You write one user-facing question for a {domain.agent_role}. "
        "Return JSON only. "
        f"The speaker is a real {domain.user_role}. "
        f"The scenario is: {domain.scenario_description}. "
        "The user knows nothing about the database, schema, tools, joins, anchors, or internal data model. "
        "The question must sound like a natural end-user request, not a schema lookup instruction. "
        "Prefer concrete natural phrasing over abstract words like related, linked, connected, field, value, or item. "
        "Use the provided entity and intent context to infer what the user is asking about, but never expose internal structure. "
        "If the anchor entity is customer/user/account/member/profile-like, use first-person phrasing naturally "
        "(for example, 'my address' or 'my account') instead of awkward compounds like 'my customer address'. "
        "If the style guide gives a preferred subject reference, use that instead of awkward account/profile wording. "
        "If the style guide gives banned phrases, do not use them or close paraphrases. "
        "The question family must shape the request: status lookup asks for one concrete value, "
        "causal chain asks about a downstream real-world detail, aggregate verification asks about a count or total of concrete items, "
        "and timeline resolution asks when something happened. "
        "If the style guide includes an answer concept reference, phrase the request around that user-facing concept "
        "rather than talking about a raw field like a title, name, label, or code value. "
        "If the style guide includes a count phrase reference, prefer that concept or a natural paraphrase when the raw target label feels awkward. "
        "If the style guide includes a count target reference and count unit hint, use them to form a natural counting phrase "
        "(for example, people -> how many people / 몇 명, cases -> how many cases / 몇 건) instead of generic item-count wording. "
        "If writing in Korean, do not leave raw English schema tokens like city, status, language, or count in the final question "
        "unless they are naturally used as end-user product terms. Prefer natural Korean wording. "
        "Do not mention tables, columns, joins, paths, anchors, records, IDs, internal tool names, or database terminology. "
        "Do not reveal the answer. "
        "Keep it concise and natural."
    )
    user_prompt = json.dumps(
        {
            "task": "write_question",
            "domain": {
                "name": domain.name,
                "language": domain.language,
                "user_role": domain.user_role,
                "agent_role": domain.agent_role,
                "scenario_description": domain.scenario_description,
            },
            "instruction": language_instruction,
            "question_contract": {
                "question_family": task.question_family,
                "outcome_type": task.outcome_type,
                "label_tier": task.label_tier,
                "tool_level": task.tool_level,
                "answer_fields": [
                    {
                        "name": field.name,
                        "type": field.type,
                        "description": field.description,
                    }
                    for field in task.answer_schema.fields
                ],
            },
            "question_context": question_context,
            "question_style_guide": style_guide,
            "fallback_seed_question": task.question,
            "quality_examples": _quality_examples(
                domain=domain,
                task=task,
                question_context=question_context,
            ),
            "response_schema": {"question": "one natural-language question"},
            "revision_feedback": feedback,
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _coherence_prompt(
    *,
    domain: DomainConfig,
    task: TaskSpec,
    question: str,
    question_context: dict[str, object],
) -> list[dict[str, str]]:
    system_prompt = (
        "You validate whether a generated user question is semantically compatible with a structured answer contract. "
        "Return JSON only."
    )
    user_prompt = json.dumps(
        {
            "task": "validate_question_coherence",
            "domain": {
                "user_role": domain.user_role,
                "agent_role": domain.agent_role,
                "scenario_description": domain.scenario_description,
                "language": domain.language,
            },
            "question": question,
            "question_contract": {
                "question_family": task.question_family,
                "outcome_type": task.outcome_type,
                "answer_fields": [
                    {
                        "name": field.name,
                        "type": field.type,
                        "description": field.description,
                    }
                    for field in task.answer_schema.fields
                ],
            },
            "question_context": question_context,
            "criteria": [
                "Pass only if the question can be answered directly by the provided answer schema.",
                "Fail if the question asks for a different concept than the answer fields imply.",
                "Fail if the question asks for extra reasoning, explanation, planning, recommendations, or actions not contained in the answer schema.",
                "Fail if the question would require hidden assumptions or information that is not represented by the answer contract.",
                "Fail if the question sounds unnatural for the configured user role, especially awkward phrases like 'my customer info' or schema-restatement wording.",
            ],
            "response_schema": {
                "pass_coherence": True,
                "reason": "brief reason",
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def generate_task_question(
    *,
    provider: ProviderConfig,
    model_ref: ModelRef,
    domain: DomainConfig,
    path: PathSpec,
    task: TaskSpec,
    question_context: dict[str, object],
    feedback: str | None = None,
    temperature: float = 1.0,
    validation_temperature: float = 0.0,
) -> str:
    del path  # question synthesis should not depend on raw schema chain text.
    try:
        generation_response = await request_json_chat_completion(
            provider=provider,
            model_ref=model_ref,
            messages=_question_prompt(
                domain=domain,
                task=task,
                question_context=question_context,
                feedback=feedback,
            ),
            temperature=temperature,
        )
    except JsonChatCompletionError as exc:
        raise QuestionGenerationError(str(exc)) from exc

    try:
        parsed = _parse_question_response(generation_response["content"])
    except (ValidationError, QuestionGenerationError) as exc:
        raise QuestionGenerationError("Question synthesis response did not match expected schema") from exc
    question = parsed.question.strip()
    if not question:
        raise QuestionGenerationError("Question synthesis returned empty question")

    try:
        validation_response = await request_json_chat_completion(
            provider=provider,
            model_ref=model_ref,
            messages=_coherence_prompt(
                domain=domain,
                task=task,
                question=question,
                question_context=question_context,
            ),
            temperature=validation_temperature,
        )
    except JsonChatCompletionError as exc:
        raise QuestionGenerationError(f"Question coherence validation failed: {exc}") from exc
    try:
        coherence = _parse_question_coherence_response(validation_response["content"])
    except (ValidationError, QuestionGenerationError) as exc:
        raise QuestionGenerationError(
            "Question coherence validation did not match expected schema"
        ) from exc
    if not coherence.pass_coherence:
        raise QuestionGenerationError(
            f"Question coherence validation rejected candidate: {coherence.reason or 'no reason'}"
        )
    return question


def _parse_question_response(content: str) -> QuestionResponse:
    try:
        payload: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise QuestionGenerationError("Question synthesis response was not valid JSON") from exc
    if "question" not in payload and "text" in payload:
        payload = {"question": payload["text"]}
    return QuestionResponse.model_validate(payload)


def _parse_question_coherence_response(content: str) -> QuestionCoherenceResponse:
    try:
        payload: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise QuestionGenerationError(
            "Question coherence validation response was not valid JSON"
        ) from exc
    return QuestionCoherenceResponse.model_validate(payload)
