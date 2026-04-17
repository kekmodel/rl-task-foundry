"""Per-conversation context for one synthesis attempt.

Synthesis backends are stateless — every call to ``run_synthesis`` carries its
own ``SynthesisConversation`` so the same backend instance can serve many
conversations concurrently. Per-conversation state (the submit_draft
controller, the pre-built composer FunctionTools for this conversation, and
the shuffle seed used to randomize tool result ordering) lives here, not on
the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rl_task_foundry.synthesis.submit_draft_tool import SubmitDraftController

if TYPE_CHECKING:
    from agents import FunctionTool


@dataclass(frozen=True, slots=True)
class SynthesisConversation:
    controller: SubmitDraftController
    sdk_tools: list["FunctionTool"]
    shuffle_seed: str
