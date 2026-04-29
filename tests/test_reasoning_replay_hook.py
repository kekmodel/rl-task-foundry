from __future__ import annotations

from agents.models.reasoning_content_replay import (
    ReasoningContentReplayContext,
    ReasoningContentSource,
)

from rl_task_foundry.infra.sdk_helpers import (
    build_reasoning_replay_hook,
)


def _ctx(target_model: str, *, origin_model: str | None = None) -> ReasoningContentReplayContext:
    return ReasoningContentReplayContext(
        model=target_model,
        base_url="https://opencode.ai/zen/v1",
        reasoning=ReasoningContentSource(
            item={"type": "reasoning", "summary": [{"type": "summary_text", "text": "..."}]},
            origin_model=origin_model,
            provider_data={},
        ),
    )


def test_hook_replays_reasoning_for_qwen_target() -> None:
    hook = build_reasoning_replay_hook()

    assert hook(_ctx("qwen3.5-plus")) is True


def test_hook_is_case_insensitive_for_qwen_target() -> None:
    hook = build_reasoning_replay_hook()

    assert hook(_ctx("QWEN3.5-PLUS")) is True
    assert hook(_ctx("Qwen3.5-Max")) is True


def test_hook_replays_reasoning_for_kimi_target() -> None:
    hook = build_reasoning_replay_hook()

    assert hook(_ctx("moonshotai/kimi-k2.5", origin_model="moonshotai/kimi-k2.5")) is True
    assert hook(_ctx("Kimi-K2.5")) is True


def test_hook_skips_replay_for_gpt_target() -> None:
    hook = build_reasoning_replay_hook()

    assert hook(_ctx("gpt-5.4-nano")) is False
    assert hook(_ctx("gpt-5.4-mini", origin_model="gpt-5.4-mini")) is False


def test_hook_preserves_deepseek_replay_via_default_fallback() -> None:
    hook = build_reasoning_replay_hook()

    assert (
        hook(_ctx("deepseek-v3", origin_model="deepseek-v3"))
        is True
    )


def test_hook_rejects_deepseek_origin_on_non_deepseek_target() -> None:
    hook = build_reasoning_replay_hook()

    assert hook(_ctx("claude-sonnet-4-6", origin_model="deepseek-v3")) is False
