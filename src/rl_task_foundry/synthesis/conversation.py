"""Per-conversation context for one synthesis attempt.

Synthesis backends are stateless — every call to ``run_synthesis`` carries its
own ``SynthesisConversation`` so the same backend instance can serve many
conversations concurrently. Per-conversation state (the submit_draft
controller, the bound atomic-tool definitions and executors, and the shuffle
seed used to randomize tool result ordering) lives here, not on the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_task_foundry.synthesis.submit_draft_tool import SubmitDraftController
from rl_task_foundry.synthesis.tool_runtime import ToolExecutor


@dataclass(frozen=True, slots=True)
class SynthesisConversation:
    controller: SubmitDraftController
    tool_definitions: list[dict[str, Any]]
    tool_executors: dict[str, ToolExecutor]
    shuffle_seed: str
