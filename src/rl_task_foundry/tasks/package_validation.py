"""Judge-agent based validation for composed task packages."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rl_task_foundry.config.models import DomainConfig, ModelRef, ProviderConfig
from rl_task_foundry.infra.json_chat_client import (
    JsonChatCompletionError,
    request_json_chat_completion,
)
from rl_task_foundry.tasks.models import PresentedToolBundle, TaskSpec


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskPackageJudgeError(RuntimeError):
    """Raised when rubric-based package validation cannot complete."""


class TaskPackageJudgeResult(StrictModel):
    pass_validation: bool
    criterion_scores: dict[str, int] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    summary: str = ""


def _validation_prompt(
    *,
    domain: DomainConfig,
    task: TaskSpec,
    presented_bundle: PresentedToolBundle,
    question_context: dict[str, object],
) -> list[dict[str, str]]:
    system_prompt = (
        "You validate whether a synthesized end-user task package is suitable for dataset generation. "
        "Return JSON only. Apply the rubric strictly."
    )
    user_prompt = json.dumps(
        {
            "task": "validate_task_package",
            "domain": {
                "name": domain.name,
                "language": domain.language,
                "user_role": domain.user_role,
                "agent_role": domain.agent_role,
                "scenario_description": domain.scenario_description,
            },
            "task_package": {
                "question": task.question,
                "question_family": task.question_family,
                "outcome_type": task.outcome_type,
                "answer_schema": [
                    {
                        "name": field.name,
                        "type": field.type,
                        "description": field.description,
                    }
                    for field in task.answer_schema.fields
                ],
                "tool_bundle": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "kind": tool.kind,
                        "output_fields": tool.output_fields,
                    }
                    for tool in presented_bundle.tools
                ],
                "question_context": question_context,
            },
            "rubric": [
                {
                    "key": "natural_language",
                    "criterion": (
                        "The question sounds like a natural request from the configured user role to the configured "
                        "agent role. Reject awkward phrases such as 'my customer info', 'linked value', 'connected item', "
                        "schema-restatement wording, or raw English schema tokens leaking into a Korean user request."
                    )
                },
                {
                    "key": "no_answer_leak",
                    "criterion": "The question does not reveal or strongly hint at the answer."
                },
                {
                    "key": "no_schema_exposure",
                    "criterion": (
                        "The user question itself does not expose schema, database, join, record, path, or ID structure. "
                        "Do not fail this criterion only because the provided tool names or descriptions are technical; "
                        "the tool bundle is for answerability checking, while schema-exposure judgment should focus on the question text."
                    )
                },
                {
                    "key": "semantic_coherence",
                    "criterion": (
                        "The question is semantically answerable by exactly the provided answer schema and matches the "
                        "intended question family. For example, a status question must ask for a status-like value, a yes/no presence check, "
                        "or a compact set of current details that align with the answer fields, "
                        "a count question must ask for a count-like value, and a causal-chain question must still read "
                        "like a practical downstream user request rather than an internal traversal. For causal-chain, "
                        "it is acceptable to ask which concrete downstream item, entity, title, language, provider, or "
                        "other real-world detail a user's payment, rental, order, or request ultimately refers to."
                    )
                },
                {
                    "key": "tool_answerability",
                    "criterion": "A capable solver with the provided tools could answer the question."
                },
            ],
            "scoring": {
                "0": "fail",
                "1": "pass",
            },
            "decision_rule": (
                "pass_validation must be true only if every rubric item passes. "
                "If any item fails, include the failure key and a short reason."
            ),
            "response_schema": {
                "pass_validation": True,
                "criterion_scores": {
                    "natural_language": 1,
                    "no_answer_leak": 1,
                    "no_schema_exposure": 1,
                    "semantic_coherence": 1,
                    "tool_answerability": 1,
                },
                "failures": ["semantic_coherence: asks for a location but answer schema is status"],
                "summary": "brief overall judgment",
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def judge_task_package(
    *,
    provider: ProviderConfig,
    model_ref: ModelRef,
    domain: DomainConfig,
    task: TaskSpec,
    presented_bundle: PresentedToolBundle,
    question_context: dict[str, object],
    temperature: float = 0.0,
) -> TaskPackageJudgeResult:
    try:
        response = await request_json_chat_completion(
            provider=provider,
            model_ref=model_ref,
            messages=_validation_prompt(
                domain=domain,
                task=task,
                presented_bundle=presented_bundle,
                question_context=question_context,
            ),
            temperature=temperature,
        )
    except JsonChatCompletionError as exc:
        raise TaskPackageJudgeError(str(exc)) from exc

    try:
        return _parse_task_package_judge_response(response["content"])
    except (ValidationError, TaskPackageJudgeError) as exc:
        raise TaskPackageJudgeError(
            "Task package judge response did not match expected schema"
        ) from exc


def _parse_task_package_judge_response(content: str) -> TaskPackageJudgeResult:
    try:
        payload: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TaskPackageJudgeError("Task package judge response was not valid JSON") from exc
    return TaskPackageJudgeResult.model_validate(payload)
