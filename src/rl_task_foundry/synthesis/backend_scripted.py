"""Deterministic scripted composer backend.

Replays a fixed sequence of atomic tool calls on the conversation
controller and submits a pre-built `SubmitDraftPayload`. Used by the
proof smoke fixture and by unit tests that need to exercise the
synthesis runtime without hitting a model provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rl_task_foundry.schema.profiler import DataProfile
from rl_task_foundry.synthesis.backend_openai_agents import (
    SynthesisConversationResult,
)
from rl_task_foundry.synthesis.conversation import SynthesisConversation
from rl_task_foundry.synthesis.submit_draft_tool import SubmitDraftPayload


@dataclass(frozen=True, slots=True)
class ScriptedAtomicToolCall:
    tool_name: str
    params: Mapping[str, object]
    result: object


@dataclass(frozen=True, slots=True)
class ScriptedComposerScript:
    atomic_tool_calls: Sequence[ScriptedAtomicToolCall]
    submit_payload: SubmitDraftPayload
    final_output_text: str = "scripted"
    turn_count: int = 1


@dataclass(slots=True)
class ScriptedComposerBackend:
    """Synthesis backend that replays a fixed script on the controller.

    Conforms structurally to ``runtime._SynthesisBackendProtocol``: exposes
    ``provider_name`` / ``model_name`` and implements ``run_synthesis``. The
    ``provider_name`` must appear in ``AppConfig.providers`` because the
    runtime indexes its circuit breakers by provider name.
    """

    script: ScriptedComposerScript
    provider_name: str
    model_name: str = "scripted"
    _last_max_turns: int | None = field(default=None, init=False, repr=False)

    async def run_synthesis(
        self,
        *,
        conversation: SynthesisConversation,
        db_id: str,
        requested_topic: str | None,
        domain_name: str,
        task_language: str,
        scenario_description: str,
        schema_summary: Mapping[str, object],
        anchor_hint: Mapping[str, object] | None = None,
        data_profile: DataProfile | None = None,
        examples_pack: object | None = None,
        affordance_map: Mapping[str, object] | None = None,
        max_turns: int,
    ) -> SynthesisConversationResult:
        del (
            requested_topic,
            domain_name,
            task_language,
            scenario_description,
            schema_summary,
            anchor_hint,
            data_profile,
            examples_pack,
            affordance_map,
        )
        self._last_max_turns = max_turns
        controller = conversation.controller
        tool_names: list[str] = []
        for call in self.script.atomic_tool_calls:
            controller.record_atomic_tool_call(
                tool_name=call.tool_name,
                params=dict(call.params),
                result=call.result,
            )
            tool_names.append(call.tool_name)
        await controller.submit(self.script.submit_payload)
        tool_names.append("submit_draft")
        return SynthesisConversationResult(
            provider=self.provider_name,
            model=self.model_name,
            final_output_text=self.script.final_output_text,
            turn_count=self.script.turn_count,
            token_usage={"requests": 0},
            tool_calls=tuple(tool_names),
        )


__all__ = [
    "ScriptedAtomicToolCall",
    "ScriptedComposerBackend",
    "ScriptedComposerScript",
]
