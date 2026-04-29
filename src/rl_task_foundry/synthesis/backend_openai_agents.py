"""OpenAI Agents SDK-backed backend for the single synthesis agent."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from time import perf_counter
from types import SimpleNamespace
from typing import Any, ClassVar

from rl_task_foundry.config.models import ModelRef, ProviderConfig, SynthesisRuntimeConfig
from rl_task_foundry.infra.sdk_helpers import (
    build_reasoning_preserving_chat_completions_model,
    build_reasoning_replay_hook,
    tool_choice_for_model,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_raw_reasoning_records as _extract_raw_reasoning_records,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_run_error_items as _extract_run_error_items,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_token_usage as _extract_token_usage,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_tool_call_name as _extract_tool_call_name,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_turn_count as _extract_turn_count,
)
from rl_task_foundry.infra.sdk_helpers import (
    load_sdk_components as _shared_load_sdk_components,
)
from rl_task_foundry.infra.sdk_helpers import (
    resolve_provider_api_key as _resolve_provider_api_key,
)
from rl_task_foundry.infra.sdk_helpers import (
    summarize_run_item as _summarize_run_item,
)
from rl_task_foundry.schema.profiler import DataProfile
from rl_task_foundry.synthesis.conversation import SynthesisConversation
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_agent_instructions,
    build_synthesis_input,
)
from rl_task_foundry.synthesis.submit_draft_tool import (
    build_submit_draft_sdk_tool,
)


@dataclass(frozen=True, slots=True)
class SynthesisConversationResult:
    provider: str
    model: str
    final_output_text: str
    turn_count: int
    token_usage: dict[str, int]
    tool_calls: tuple[str, ...] = ()


def _final_output_text(run_result: object) -> str:
    final_output = getattr(run_result, "final_output", None)
    return final_output if isinstance(final_output, str) else str(final_output or "")


def _tool_calls_from_result(run_result: object) -> tuple[str, ...]:
    return tuple(
        tool_name
        for item in getattr(run_result, "new_items", []) or []
        if (tool_name := _extract_tool_call_name(item)) is not None
    )


def _tool_output_text(tool_result: object) -> str | None:
    output = getattr(tool_result, "output", None)
    return output if isinstance(output, str) else None


def _is_submit_draft_required_output(output: str) -> bool:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("error") == "submit_draft_required"


def _add_token_usage(
    total: dict[str, int],
    usage: dict[str, int],
) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + int(value)


def _solver_rollout_timeout_allowance_s(config: object | None) -> float:
    """Extra wall-clock budget for submit_draft's internal solver rollout.

    The SDK runner timeout wraps tool execution too. Since submit_draft runs
    the solver rollout inside a tool call, the composer timeout must leave
    enough room for infra-failure replacement solver attempts.
    """

    if config is None:
        return 0.0
    calibration = getattr(config, "calibration", None)
    database = getattr(config, "database", None)
    solver_runtime = getattr(config, "solver_runtime", None)
    if calibration is None or database is None or solver_runtime is None:
        return 0.0

    target_evaluable_runs = int(getattr(calibration, "max_solver_runs", 0) or 0)
    if target_evaluable_runs <= 0:
        return 0.0
    batch_size = max(
        1,
        min(
            int(getattr(calibration, "solver_batch_size", 1) or 1),
            target_evaluable_runs,
        ),
    )
    max_attempts = max(target_evaluable_runs, target_evaluable_runs * 2)
    max_batches = math.ceil(max_attempts / batch_size)
    episode_timeout_s = (
        int(getattr(database, "statement_timeout_ms", 0) or 0)
        * int(getattr(solver_runtime, "max_turns", 0) or 0)
        / 1000.0
    )
    if episode_timeout_s <= 0:
        return 0.0
    return max_batches * episode_timeout_s


def _persist_reasoning_records(
    *,
    event_logger: object | None,
    records: list[dict[str, Any]],
    provider: str,
    model: str,
    segment_index: int,
) -> tuple[str | None, int]:
    if not records:
        return None, 0
    writer = getattr(event_logger, "write_sidecar_jsonl", None)
    if not callable(writer):
        return None, len(records)
    enriched = [
        {
            "actor": "composer",
            "actor_id": None,
            "provider": provider,
            "model": model,
            "segment_index": segment_index,
            **record,
        }
        for record in records
    ]
    path = writer("reasoning_content.jsonl", enriched)
    return (str(path) if path is not None else None), len(records)


def _append_feedback_input(
    run_result: object,
    feedback_message: str,
) -> list[object] | None:
    to_input_list = getattr(run_result, "to_input_list", None)
    if not callable(to_input_list):
        return None
    continuation = list(to_input_list(mode="preserve_all"))
    continuation.append({"role": "user", "content": feedback_message})
    return continuation


def _controller_accepts_more_feedback(controller: object) -> bool:
    submissions_left = getattr(controller, "submissions_left", None)
    if not callable(submissions_left):
        return True
    return int(submissions_left()) > 0


def _build_agent(
    sdk: SimpleNamespace,
    *,
    model: object,
    model_name: str,
    tools: list[object],
    tool_use_behavior: object,
    runtime_config: SynthesisRuntimeConfig,
) -> object:
    return sdk.Agent(
        name="synthesis",
        instructions=build_synthesis_agent_instructions(runtime_config),
        model=model,
        tools=tools,
        output_type=None,
        tool_use_behavior=tool_use_behavior,
        model_settings=sdk.ModelSettings(
            parallel_tool_calls=False,
            tool_choice=tool_choice_for_model(model_name),
        ),
        reset_tool_choice=False,
    )


@dataclass(slots=True)
class OpenAIAgentsSynthesisBackend:
    """Stateless OpenAI Agents synthesis backend.

    All per-conversation state (controller, atomic-tool definitions and
    executors) is passed in as a ``SynthesisConversation`` argument to
    ``run_synthesis``. Multiple conversations may share one backend instance
    concurrently; the only mutable state on the backend is the cached SDK
    components and the (class-level) shared OpenAI client/model cache.
    """

    model_ref: ModelRef
    provider_config: ProviderConfig
    runtime_config: SynthesisRuntimeConfig
    _sdk: SimpleNamespace | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _shared_models: ClassVar[dict[tuple[int, int, str, str | None, str, float, str], object]] = {}

    @property
    def provider_name(self) -> str:
        return self.model_ref.provider

    @property
    def model_name(self) -> str:
        return self.model_ref.model

    @classmethod
    def clear_model_cache(cls) -> None:
        cls._shared_models.clear()

    def _sdk_components(self) -> SimpleNamespace:
        if self._sdk is None:
            self._sdk = _shared_load_sdk_components(include_tools_to_final_output=True)
        return self._sdk

    def _build_model(self, sdk: SimpleNamespace) -> object:
        if self._model is not None:
            return self._model
        if self.provider_config.type not in {"openai", "openai_compatible"}:
            raise NotImplementedError(
                "OpenAI Agents synthesis backend only supports openai/openai_compatible providers"
            )
        api_key = _resolve_provider_api_key(self.provider_config)
        cache_key = (
            id(sdk.AsyncOpenAI),
            id(sdk.OpenAIChatCompletionsModel),
            self.provider_config.type,
            self.provider_config.base_url,
            api_key,
            float(self.provider_config.timeout_s),
            self.model_ref.model,
        )
        cached = self._shared_models.get(cache_key)
        if cached is not None:
            self._model = cached
            return self._model
        client = sdk.AsyncOpenAI(
            api_key=api_key,
            base_url=self.provider_config.base_url,
            timeout=float(self.provider_config.timeout_s),
            max_retries=self.provider_config.max_retries,
        )
        self._model = build_reasoning_preserving_chat_completions_model(
            sdk,
            model=self.model_ref.model,
            openai_client=client,
            should_replay_reasoning_content=build_reasoning_replay_hook(),
        )
        self._shared_models[cache_key] = self._model
        return self._model

    def _build_tools(self, conversation: SynthesisConversation) -> list[object]:
        sdk_tools: list[object] = list(conversation.sdk_tools)
        sdk_tools.append(build_submit_draft_sdk_tool(conversation.controller))
        return sdk_tools

    def _build_tool_use_behavior(
        self,
        sdk: SimpleNamespace,
        conversation: SynthesisConversation,
    ) -> object:
        controller = conversation.controller

        def _finalize_on_submit(_context_wrapper: object, tool_results: list[object]) -> object:
            for tool_result in tool_results:
                tool_name = getattr(getattr(tool_result, "tool", None), "name", None)
                output = _tool_output_text(tool_result)
                if (
                    tool_name != "submit_draft"
                    and output is not None
                    and _is_submit_draft_required_output(output)
                ):
                    return sdk.ToolsToFinalOutputResult(
                        is_final_output=True,
                        final_output=output,
                    )
                if tool_name != "submit_draft":
                    continue
                if output is None:
                    continue
                normalized = output.strip()
                if (
                    normalized.startswith("Accepted:")
                    or "BudgetExhaustedError: No more attempts." in normalized
                    or controller._terminated_too_hard
                ):
                    return sdk.ToolsToFinalOutputResult(
                        is_final_output=True,
                        final_output=normalized,
                    )
            return sdk.ToolsToFinalOutputResult(is_final_output=False, final_output=None)

        return _finalize_on_submit

    async def run_synthesis(
        self,
        *,
        conversation: SynthesisConversation,
        db_id: str,
        requested_topic: str | None,
        domain_name: str,
        task_language: str,
        scenario_description: str,
        schema_summary: dict[str, object],
        tool_surface_summary: dict[str, object],
        max_turns: int,
        anchor_hint: dict[str, object] | None = None,
        data_profile: DataProfile | None = None,
        examples_pack: object | None = None,
        affordance_map: dict[str, object] | None = None,
    ) -> SynthesisConversationResult:
        sdk = self._sdk_components()
        if hasattr(sdk, "set_tracing_disabled"):
            sdk.set_tracing_disabled(
                disabled=(not self.runtime_config.tracing) or self.provider_config.type != "openai"
            )
        model = self._build_model(sdk)
        tools = self._build_tools(conversation)
        agent = _build_agent(
            sdk,
            model=model,
            model_name=self.model_ref.model,
            tools=tools,
            tool_use_behavior=self._build_tool_use_behavior(sdk, conversation),
            runtime_config=self.runtime_config,
        )
        request_input = build_synthesis_input(
            domain_name=domain_name,
            scenario_description=scenario_description,
            requested_topic=requested_topic,
            task_language=task_language,
            schema_summary=schema_summary,
            tool_surface_summary=tool_surface_summary,
            runtime_config=self.runtime_config,
            anchor_hint=anchor_hint,
            data_profile=data_profile,
            examples_pack=examples_pack,
            affordance_map=affordance_map,
        )
        started_at = perf_counter()
        solver_rollout_timeout_allowance_s = _solver_rollout_timeout_allowance_s(
            getattr(conversation.controller, "config", None)
        )
        deadline = (
            started_at
            + self.runtime_config.run_timeout_s
            + solver_rollout_timeout_allowance_s
        )
        run_input: str | list[object] = request_input
        total_turn_count = 0
        token_usage: dict[str, int] = {}
        tool_calls: list[str] = []
        run_item_summaries: list[dict[str, Any]] = []
        protocol_feedback_events = 0
        reasoning_content_path: str | None = None
        reasoning_content_items = 0
        segment_index = 0
        final_output_text = ""
        while True:
            segment_index += 1
            turns_left = max(1, max_turns - total_turn_count)
            try:
                remaining_timeout_s = deadline - perf_counter()
                if remaining_timeout_s <= 0:
                    raise TimeoutError(
                        "synthesis run exceeded run_timeout_s="
                        f"{self.runtime_config.run_timeout_s}"
                    )
                try:
                    run_result = await asyncio.wait_for(
                        sdk.Runner.run(
                            agent,
                            run_input,
                            max_turns=turns_left,
                            session=None,
                        ),
                        timeout=remaining_timeout_s,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        "synthesis run exceeded run_timeout_s="
                        f"{self.runtime_config.run_timeout_s}"
                    ) from exc
            except Exception as exc:
                latency_ms = int((perf_counter() - started_at) * 1000)
                event_logger = getattr(conversation.controller, "event_logger", None)
                error_reasoning_path, error_reasoning_items = _persist_reasoning_records(
                    event_logger=event_logger,
                    records=_extract_raw_reasoning_records(
                        getattr(getattr(exc, "run_data", None), "new_items", []) or []
                    ),
                    provider=self.provider_name,
                    model=self.model_name,
                    segment_index=segment_index,
                )
                if error_reasoning_path is not None:
                    reasoning_content_path = error_reasoning_path
                reasoning_content_items += error_reasoning_items
                if event_logger is not None:
                    event_logger.log_sync(
                        actor="composer",
                        event_type="synthesis_failed",
                        payload={
                            "error_type": exc.__class__.__name__,
                            "error_detail": str(exc),
                            "latency_ms": latency_ms,
                            "max_turns": max_turns,
                            "turn_count": total_turn_count,
                            "token_usage": token_usage,
                            "tool_calls": list(tool_calls),
                            "protocol_feedback_events": protocol_feedback_events,
                            "solver_rollout_timeout_allowance_s": (
                                solver_rollout_timeout_allowance_s
                            ),
                            "reasoning_content_path": reasoning_content_path,
                            "reasoning_content_items": reasoning_content_items,
                            "run_items": [
                                *run_item_summaries,
                                *_extract_run_error_items(exc),
                            ],
                        },
                    )
                raise
            segment_turn_count = max(1, _extract_turn_count(run_result))
            total_turn_count += segment_turn_count
            segment_tool_calls = _tool_calls_from_result(run_result)
            tool_calls.extend(segment_tool_calls)
            _add_token_usage(token_usage, _extract_token_usage(run_result))
            run_item_summaries.extend(
                _summarize_run_item(item)
                for item in getattr(run_result, "new_items", []) or []
            )
            segment_reasoning_path, segment_reasoning_items = _persist_reasoning_records(
                event_logger=getattr(conversation.controller, "event_logger", None),
                records=_extract_raw_reasoning_records(
                    getattr(run_result, "new_items", []) or []
                ),
                provider=self.provider_name,
                model=self.model_name,
                segment_index=segment_index,
            )
            if segment_reasoning_path is not None:
                reasoning_content_path = segment_reasoning_path
            reasoning_content_items += segment_reasoning_items
            final_output_text = _final_output_text(run_result)

            controller = conversation.controller
            missing_submit = (
                getattr(controller, "accepted_draft", None) is None
                and (not tool_calls or tool_calls[-1] != "submit_draft")
            )
            if not missing_submit:
                break
            record_feedback = getattr(controller, "record_missing_submit_feedback", None)
            if not callable(record_feedback):
                break
            if not _controller_accepts_more_feedback(controller):
                break
            feedback_message = record_feedback(
                final_output_text=final_output_text,
                tool_calls=tuple(tool_calls),
            )
            protocol_feedback_events += 1
            if (
                "BudgetExhaustedError: No more attempts." in feedback_message
                or total_turn_count >= max_turns
                or not _controller_accepts_more_feedback(controller)
            ):
                break
            continuation_input = _append_feedback_input(run_result, feedback_message)
            if continuation_input is None:
                break
            run_input = continuation_input

        latency_ms = int((perf_counter() - started_at) * 1000)
        event_logger = getattr(conversation.controller, "event_logger", None)
        if event_logger is not None:
            event_logger.log_sync(
                actor="composer",
                event_type="synthesis_completed",
                payload={
                    "final_output_text": final_output_text,
                    "latency_ms": latency_ms,
                    "turn_count": total_turn_count,
                    "token_usage": token_usage,
                    "tool_calls": list(tool_calls),
                    "protocol_feedback_events": protocol_feedback_events,
                    "solver_rollout_timeout_allowance_s": (
                        solver_rollout_timeout_allowance_s
                    ),
                    "reasoning_content_path": reasoning_content_path,
                    "reasoning_content_items": reasoning_content_items,
                    "run_items": run_item_summaries,
                },
            )
        return SynthesisConversationResult(
            provider=self.provider_name,
            model=self.model_name,
            final_output_text=final_output_text,
            turn_count=total_turn_count,
            token_usage=token_usage,
            tool_calls=tuple(tool_calls),
        )
