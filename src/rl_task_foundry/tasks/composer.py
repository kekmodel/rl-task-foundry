"""Task package composition interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Literal

from rl_task_foundry.config.models import DomainConfig, ModelRef, ProviderConfig
from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.path_catalog import PathSpec
from rl_task_foundry.tasks.question_generation import (
    QuestionGenerationError,
    generate_task_question,
)
from rl_task_foundry.tasks.package_validation import (
    TaskPackageJudgeError,
    judge_task_package,
)
from rl_task_foundry.tasks.models import (
    PresentedToolBundle,
    PresentedToolSpec,
    TaskPackage,
    TaskSpec,
)
from rl_task_foundry.tools.naming_eval import evaluate_tool_bundle_naming
from rl_task_foundry.tools.model_naming import ToolNamingGenerationError, generate_named_tool_bundle
from rl_task_foundry.tools.models import ToolBundle


@dataclass(slots=True)
class ComposeRequest:
    graph: SchemaGraph
    task: TaskSpec
    path: PathSpec
    canonical_bundle: ToolBundle
    source_canonical_bundle_id: str | None = None
    question_context: dict[str, object] = field(default_factory=dict)
    fallback_presented_bundle: ToolBundle | None = None
    distractor_bundles: list[ToolBundle] = field(default_factory=list)


class TaskComposer:
    """Compose a task package from a task skeleton and canonical tool bundle."""

    def __init__(
        self,
        *,
        domain: DomainConfig,
        provider: ProviderConfig | None = None,
        model_ref: ModelRef | None = None,
        question_temperature: float = 1.0,
        question_validation_temperature: float = 0.0,
        naming_temperature_l2: float = 1.0,
    ) -> None:
        self._domain = domain
        self._provider = provider
        self._model_ref = model_ref
        self._question_temperature = question_temperature
        self._question_validation_temperature = question_validation_temperature
        self._naming_temperature_l2 = naming_temperature_l2

    async def compose(self, request: ComposeRequest) -> TaskPackage:
        focused_request = _task_focused_request(request)
        task = await self._compose_task_question(focused_request)
        presented_bundle = await self._compose_presented_bundle(
            replace(focused_request, task=task)
        )
        task, presented_bundle = await self._apply_task_package_judge(
            focused_request,
            task=task,
            presented_bundle=presented_bundle,
        )
        source_canonical_bundle_id = (
            focused_request.source_canonical_bundle_id
            or focused_request.canonical_bundle.bundle_id
        )
        base_bundle = PresentedToolBundle(
            bundle_id=f"{focused_request.canonical_bundle.bundle_id}::task::{task.task_id}::L1",
            canonical_bundle_id=source_canonical_bundle_id,
            path_id=focused_request.path.path_id,
            tool_level=1,
            question_family=task.question_family,
            outcome_type=task.outcome_type,
            tools=_present_tool(tool_bundle=focused_request.canonical_bundle, role="core"),
            generation_metadata={"presentation_strategy": "canonical_rule_based"},
        )
        task = task.model_copy(
            update={
                "presented_tool_bundle_id": presented_bundle.bundle_id,
                "provenance_requirements": _provenance_requirements(
                    task=task,
                    presented_bundle=presented_bundle,
                ),
            }
        )
        presentation_options = [base_bundle]
        if presented_bundle.tool_level != base_bundle.tool_level:
            presentation_options.append(presented_bundle)
        return TaskPackage(
            task=task,
            presented_tool_bundle=presented_bundle,
            presentation_options=presentation_options,
        )

    async def _apply_task_package_judge(
        self,
        request: ComposeRequest,
        *,
        task: TaskSpec,
        presented_bundle: PresentedToolBundle,
    ) -> tuple[TaskSpec, PresentedToolBundle]:
        if self._provider is None or self._model_ref is None or not request.question_context:
            return task, presented_bundle

        generation_context = _question_generation_context(request.question_context)
        judged_task = task
        judged_bundle = presented_bundle
        judge_runs: list[dict[str, object]] = []
        judge_feedback: str | None = None

        for judge_attempt in range(1, 3):
            result = await self._run_task_package_judge(
                task=judged_task,
                presented_bundle=judged_bundle,
                question_context=generation_context,
            )
            judge_runs.append(result)
            metadata = dict(judged_task.question_generation_metadata)
            metadata["task_package_judge"] = result
            metadata["task_package_judge_runs"] = judge_runs
            metadata["task_package_judge_attempts"] = judge_attempt
            judged_task = judged_task.model_copy(update={"question_generation_metadata": metadata})
            if result["pass_validation"] or judged_task.question_source != "model_generated":
                return judged_task, judged_bundle
            judge_feedback = _task_package_judge_feedback(result)
            if judge_attempt >= 2:
                break
            regenerated_task = await self._compose_task_question(
                request,
                feedback=judge_feedback,
            )
            if regenerated_task.question_source != "model_generated":
                break
            judged_task = regenerated_task
            judged_bundle = await self._compose_presented_bundle(
                ComposeRequest(
                    graph=request.graph,
                    task=judged_task,
                    path=request.path,
                    canonical_bundle=request.canonical_bundle,
                    question_context=request.question_context,
                    fallback_presented_bundle=request.fallback_presented_bundle,
                    distractor_bundles=request.distractor_bundles,
                )
            )

        fallback_task = request.task.model_copy(
            update={
                "question_source": "seed_fallback",
                "question_generation_metadata": {
                    **judged_task.question_generation_metadata,
                    "status": "fallback",
                    "fallback_reason": "task_package_judge_rejected",
                    "judge_feedback": judge_feedback,
                },
            }
        )
        fallback_presented_bundle = await self._compose_presented_bundle(
            ComposeRequest(
                graph=request.graph,
                task=fallback_task,
                path=request.path,
                canonical_bundle=request.canonical_bundle,
                question_context=request.question_context,
                fallback_presented_bundle=request.fallback_presented_bundle,
                distractor_bundles=request.distractor_bundles,
            )
        )
        return fallback_task, fallback_presented_bundle

    async def _run_task_package_judge(
        self,
        *,
        task: TaskSpec,
        presented_bundle: PresentedToolBundle,
        question_context: dict[str, object],
    ) -> dict[str, object]:
        try:
            result = await judge_task_package(
                provider=self._provider,
                model_ref=self._model_ref,
                domain=self._domain,
                task=task,
                presented_bundle=presented_bundle,
                question_context=question_context,
                temperature=self._question_validation_temperature,
            )
        except TaskPackageJudgeError as exc:
            return {
                "pass_validation": False,
                "criterion_scores": {},
                "failures": [f"judge_error: {exc}"],
                "summary": "task package judge failed to execute",
            }
        return result.model_dump(mode="json")

    async def _compose_task_question(
        self,
        request: ComposeRequest,
        *,
        feedback: str | None = None,
    ) -> TaskSpec:
        if self._provider is None or self._model_ref is None or not request.question_context:
            return request.task

        question = request.task.question
        metadata: dict[str, object] = {"attempts": 0}
        feedback_hint = feedback
        if feedback_hint is not None:
            metadata["external_feedback"] = feedback_hint
        generation_context = _question_generation_context(request.question_context)
        for attempt in range(1, 4):
            metadata["attempts"] = attempt
            try:
                candidate = await generate_task_question(
                    provider=self._provider,
                    model_ref=self._model_ref,
                    domain=self._domain,
                    path=request.path,
                    task=request.task.model_copy(update={"question": question}),
                    question_context=generation_context,
                    feedback=feedback_hint,
                    temperature=self._question_temperature,
                    validation_temperature=self._question_validation_temperature,
                )
            except QuestionGenerationError as exc:
                metadata["last_error"] = str(exc)
                feedback_hint = f"Previous question was invalid: {exc}. Write a more natural user request."
                continue
            violations = _question_policy_violations(candidate, request.question_context)
            if not violations:
                metadata["status"] = "accepted"
                return request.task.model_copy(
                    update={
                        "question": candidate,
                        "question_source": "model_generated",
                        "question_generation_metadata": metadata,
                    }
                )
            feedback_hint = (
                "Previous question leaked answer-like content or sounded too schematic. "
                f"Fix: {violations}"
            )
            metadata["last_error"] = feedback_hint
        metadata["status"] = "fallback"
        return request.task.model_copy(
            update={
                "question_source": "seed_fallback",
                "question_generation_metadata": metadata,
            }
        )

    async def _compose_presented_bundle(
        self,
        request: ComposeRequest,
    ) -> PresentedToolBundle:
        core_bundle = _retarget_bundle_level(
            request.canonical_bundle,
            request.task.tool_level,
        )
        source_canonical_bundle_id = (
            request.source_canonical_bundle_id or request.canonical_bundle.bundle_id
        )
        generation_metadata: dict[str, object] = {
            "canonical_bundle_id": source_canonical_bundle_id,
            "question_family": request.task.question_family,
            "tool_level": request.task.tool_level,
            "distractor_bundle_count": len(request.distractor_bundles),
        }

        if request.task.tool_level == 2 and self._provider and self._model_ref:
            fallback_eval = None
            if request.fallback_presented_bundle is not None:
                fallback_eval = evaluate_tool_bundle_naming(
                    request.graph,
                    request.path,
                    request.fallback_presented_bundle,
                )
                generation_metadata["fallback_opacity"] = round(
                    fallback_eval.schema_opacity_score,
                    3,
                )
                generation_metadata["fallback_overlap"] = round(
                    fallback_eval.raw_identifier_overlap_ratio,
                    3,
                )

            feedback: str | None = None
            best_generated_bundle: ToolBundle | None = None
            best_generated_eval = None
            last_error: str | None = None

            for attempt in range(1, 4):
                generation_metadata["generation_attempts"] = attempt
                try:
                    generated_bundle = await generate_named_tool_bundle(
                        provider=self._provider,
                        model_ref=self._model_ref,
                        domain=self._domain,
                        path=request.path,
                        bundle=core_bundle,
                        task_question=request.task.question,
                        question_family=request.task.question_family,
                        outcome_type=request.task.outcome_type,
                        feedback=feedback,
                        temperature=self._naming_temperature_l2,
                    )
                except ToolNamingGenerationError as exc:
                    last_error = str(exc)
                    feedback = f"Previous attempt failed validation: {exc}. Rewrite all tool names."
                    continue

                generated_eval = evaluate_tool_bundle_naming(
                    request.graph,
                    request.path,
                    generated_bundle,
                )
                if (
                    best_generated_eval is None
                    or generated_eval.raw_identifier_overlap_ratio
                    < best_generated_eval.raw_identifier_overlap_ratio
                ):
                    best_generated_bundle = generated_bundle
                    best_generated_eval = generated_eval

                if (
                    not generated_eval.policy_violations
                    and (
                        fallback_eval is None
                        or generated_eval.raw_identifier_overlap_ratio
                        < fallback_eval.raw_identifier_overlap_ratio
                    )
                ):
                    core_bundle = generated_bundle
                    generation_metadata["generated_opacity"] = round(
                        generated_eval.schema_opacity_score,
                        3,
                    )
                    generation_metadata["generated_overlap"] = round(
                        generated_eval.raw_identifier_overlap_ratio,
                        3,
                    )
                    generation_metadata["generated_policy_violations"] = []
                    generation_metadata["presentation_strategy"] = "task_context_model_generated"
                    break

                feedback = _quality_feedback(generated_eval)
                last_error = (
                    "model-generated naming did not pass L2 quality gate"
                    if generated_eval.policy_violations
                    else "model-generated naming did not improve over fallback"
                )
            else:
                if best_generated_eval is not None:
                    generation_metadata["generated_opacity"] = round(
                        best_generated_eval.schema_opacity_score,
                        3,
                    )
                    generation_metadata["generated_overlap"] = round(
                        best_generated_eval.raw_identifier_overlap_ratio,
                        3,
                    )
                    generation_metadata["generated_policy_violations"] = list(
                        best_generated_eval.policy_violations
                    )
                generation_metadata["presentation_strategy"] = "task_context_fallback_alias"
                generation_metadata["naming_generation_error"] = last_error or "naming quality gate failed"
                if request.fallback_presented_bundle is not None:
                    core_bundle = request.fallback_presented_bundle
                elif best_generated_bundle is not None:
                    core_bundle = best_generated_bundle
                else:
                    generation_metadata["presentation_strategy"] = "canonical_rule_based_fallback"
        elif request.task.tool_level == 2:
            generation_metadata["presentation_strategy"] = "canonical_rule_based_fallback"
        else:
            generation_metadata["presentation_strategy"] = "canonical_rule_based"

        tools = _present_tool(tool_bundle=core_bundle, role="core") + [
            presented
            for bundle in request.distractor_bundles
            for presented in _present_tool(tool_bundle=bundle, role="distractor")
        ]

        return PresentedToolBundle(
            bundle_id=f"{core_bundle.bundle_id}::task::{request.task.task_id}",
            canonical_bundle_id=source_canonical_bundle_id,
            path_id=request.path.path_id,
            tool_level=request.task.tool_level,
            question_family=request.task.question_family,
            outcome_type=request.task.outcome_type,
            tools=tools,
            generation_metadata=generation_metadata,
        )


def _present_tool(
    *,
    tool_bundle: ToolBundle,
    role: Literal["core", "distractor"],
) -> list[PresentedToolSpec]:
    return [
        PresentedToolSpec(
            name=tool.name,
            description=tool.description,
            semantic_key=tool.semantic_key,
            kind=tool.kind,
            parameter_names=[parameter.name for parameter in tool.parameters],
            output_fields=list(tool.output_fields),
            name_source=tool.name_source,
            presentation_role=role,
        )
        for tool in tool_bundle.tools
    ]


def _retarget_bundle_level(
    bundle: ToolBundle,
    tool_level: Literal[1, 2],
) -> ToolBundle:
    if bundle.tool_level == tool_level:
        return bundle
    return ToolBundle(
        bundle_id=f"{bundle.bundle_id}::presented-L{tool_level}",
        path_id=bundle.path_id,
        tool_level=tool_level,
        tools=[
            replace(tool, tool_level=tool_level)
            for tool in bundle.tools
        ],
    )


def _task_focused_request(request: ComposeRequest) -> ComposeRequest:
    source_canonical_bundle_id = (
        request.source_canonical_bundle_id or request.canonical_bundle.bundle_id
    )
    return ComposeRequest(
        graph=request.graph,
        task=request.task,
        path=request.path,
        canonical_bundle=_task_focused_core_bundle(request.task, request.canonical_bundle),
        source_canonical_bundle_id=source_canonical_bundle_id,
        question_context=request.question_context,
        fallback_presented_bundle=(
            _task_focused_core_bundle(request.task, request.fallback_presented_bundle)
            if request.fallback_presented_bundle is not None
            else None
        ),
        distractor_bundles=request.distractor_bundles,
    )


def _task_focused_core_bundle(task: TaskSpec, bundle: ToolBundle) -> ToolBundle:
    count_semantic_key = task.contract_metadata.get("count_semantic_key")
    if not (isinstance(count_semantic_key, str) and count_semantic_key):
        return bundle
    if _task_answer_shape(task) != "count":
        return bundle
    focused_tools = [
        tool
        for tool in bundle.tools
        if tool.semantic_key == count_semantic_key
    ]
    if not focused_tools:
        return bundle
    return ToolBundle(
        bundle_id=f"{bundle.bundle_id}::focus::{_focused_bundle_key_suffix(count_semantic_key)}",
        path_id=bundle.path_id,
        tool_level=bundle.tool_level,
        tools=focused_tools,
    )


def _focused_bundle_key_suffix(semantic_key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", semantic_key.lower()).strip("-")


def _quality_feedback(evaluation) -> str:
    raw_tokens = sorted(
        {
            *(token for check in evaluation.per_tool for token in check.raw_table_hits),
            *(token for check in evaluation.per_tool for token in check.raw_column_hits),
        }
    )
    sample_names = ", ".join(check.name for check in evaluation.per_tool[:4])
    parts = [
        "Previous attempt was still too literal or otherwise failed the L2 quality gate.",
        f"Violations: {evaluation.policy_violations or ['none but not better than fallback']}.",
        f"Current names: {sample_names}.",
    ]
    if raw_tokens:
        parts.append(
            "Avoid repeating these raw schema tokens directly unless absolutely necessary: "
            + ", ".join(raw_tokens)
            + "."
        )
    parts.append(
        "Rewrite every tool name so it sounds like a UI/workflow label, not a schema chain."
    )
    return " ".join(parts)


def _question_policy_violations(
    question: str,
    question_context: dict[str, object],
) -> list[str]:
    normalized_question = question.strip().lower()
    violations: list[str] = []
    if not normalized_question:
        return ["empty_question"]

    forbidden_markers = question_context.get("forbidden_markers")
    if isinstance(forbidden_markers, list):
        leaked = [
            marker.get("display", marker.get("normalized", ""))
            for marker in forbidden_markers
            if isinstance(marker, dict)
            and _text_matches_forbidden_marker(normalized_question, marker)
        ]
        if leaked:
            violations.append(f"contains forbidden answer/context terms: {sorted(set(leaked))}")
    else:
        forbidden_terms = question_context.get("forbidden_terms")
        if isinstance(forbidden_terms, list):
            leaked: list[str] = []
            for term in forbidden_terms:
                if not isinstance(term, str):
                    continue
                normalized_term = term.strip().lower()
                if len(normalized_term) < 3:
                    continue
                if normalized_term in normalized_question:
                    leaked.append(normalized_term)
            if leaked:
                violations.append(
                    f"contains forbidden answer/context terms: {sorted(set(leaked))}"
                )

    schema_hits = [
        phrase
        for phrase in ("table", "column", "join", "path", "anchor", "schema", "database", "record")
        if phrase in normalized_question
    ]
    if schema_hits:
        violations.append(f"contains internal/schema wording: {schema_hits}")
    if " id " in f" {normalized_question} ":
        violations.append("contains raw id terminology")

    if str(question_context.get("language", "")).lower() == "ko":
        raw_english_tokens = _raw_schema_tokens(question_context)
        leaked_tokens = [
            token
            for token in raw_english_tokens
            if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", normalized_question)
        ]
        if leaked_tokens:
            violations.append(f"contains raw english schema tokens in korean: {sorted(set(leaked_tokens))}")
    return violations


def _provenance_requirements(
    *,
    task: TaskSpec,
    presented_bundle: PresentedToolBundle,
) -> list[str]:
    core_tools = [tool for tool in presented_bundle.tools if tool.presentation_role == "core"]
    if not core_tools:
        return []

    answer_shape = _task_answer_shape(task)
    if answer_shape == "count":
        semantic_key = task.contract_metadata.get("count_semantic_key")
        if isinstance(semantic_key, str) and semantic_key:
            return [f"semantic_key:{semantic_key}"]
        return _semantic_key_requirements(core_tools, kind="count")
    if answer_shape == "exists":
        return _semantic_key_requirements(core_tools, kind="exists")
    if answer_shape == "list":
        return _semantic_key_requirements(core_tools, kind="list_related")
    if answer_shape == "latest_scalar":
        prefixes = sorted(
            {
                f"semantic_key_prefix:{tool.semantic_key}"
                for tool in core_tools
                if tool.kind == "timeline"
            }
        )
        if prefixes:
            return prefixes
        return ["semantic_key:__missing_timeline_tool__"]
    return _semantic_key_requirements(core_tools, kind="lookup")


def _semantic_key_requirements(
    tools: list[PresentedToolSpec],
    *,
    kind: str,
) -> list[str]:
    requirements = sorted(
        {
            f"semantic_key:{tool.semantic_key}"
            for tool in tools
            if tool.kind == kind
        }
    )
    if requirements:
        return requirements
    return sorted({f"semantic_key:{tool.semantic_key}" for tool in tools})


def _task_answer_shape(task: TaskSpec) -> str:
    if any(field.source_columns and field.source_columns[0] == "meta:count" for field in task.answer_schema.fields):
        return "count"
    if any(field.source_columns and field.source_columns[0] == "meta:exists" for field in task.answer_schema.fields):
        return "exists"
    if task.answer_schema.fields and all(field.type.startswith("list[") for field in task.answer_schema.fields):
        return "list"
    if len(task.answer_schema.fields) > 1:
        return "record"
    if task.question_family == "timeline_resolution":
        return "latest_scalar"
    return "scalar"


def _question_generation_context(question_context: dict[str, object]) -> dict[str, object]:
    sanitized = dict(question_context)
    sanitized.pop("forbidden_markers", None)
    sanitized.pop("forbidden_terms", None)
    return sanitized


def _task_package_judge_feedback(result: dict[str, object]) -> str:
    failures = result.get("failures")
    if isinstance(failures, list) and failures:
        details = "; ".join(str(item) for item in failures[:4])
    else:
        details = str(result.get("summary", "task package judge rejected the question"))
    return (
        "Rewrite the question so it sounds like a real user request and matches the answer contract exactly. "
        f"Fix these issues: {details}"
    )


def _text_matches_forbidden_marker(text: str, marker: dict[str, object]) -> bool:
    normalized = str(marker.get("normalized", "")).strip().lower()
    if not normalized:
        return False
    match_mode = str(marker.get("match_mode", "substring"))
    if match_mode == "token":
        tokens = {
            token
            for token in text.replace("/", " ").replace("-", " ").replace("_", " ").split()
            if token
        }
        return normalized in tokens
    return normalized in text


def _raw_schema_tokens(question_context: dict[str, object]) -> set[str]:
    tokens: set[str] = set()
    path_entity_labels = question_context.get("path_entity_labels")
    if isinstance(path_entity_labels, list):
        for label in path_entity_labels:
            if not isinstance(label, str):
                continue
            tokens.update(_ascii_tokens(label))
    answer_fields = question_context.get("answer_fields")
    if isinstance(answer_fields, list):
        for field in answer_fields:
            if not isinstance(field, dict):
                continue
            for key in ("name", "label"):
                value = field.get(key)
                if isinstance(value, str):
                    tokens.update(_ascii_tokens(value))
    return {token for token in tokens if len(token) >= 4}


def _ascii_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]+", value.strip().lower())
        if token
    }
