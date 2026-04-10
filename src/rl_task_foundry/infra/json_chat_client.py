"""Shared OpenAI-compatible JSON chat client helper."""

from __future__ import annotations

import os
from typing import Any

import httpx

from rl_task_foundry.config.models import ModelRef, ProviderConfig


class JsonChatCompletionError(RuntimeError):
    """Raised when a JSON chat completion request fails."""


def resolve_provider_api_key(provider: ProviderConfig) -> str:
    env_value = os.environ.get(provider.api_key_env)
    if env_value:
        return env_value
    if provider.type == "openai_compatible":
        return "dummy"
    raise JsonChatCompletionError(f"Missing API key env var: {provider.api_key_env}")


async def request_json_chat_completion(
    *,
    provider: ProviderConfig,
    model_ref: ModelRef,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict[str, Any]:
    if provider.type not in {"openai", "openai_compatible"}:
        raise JsonChatCompletionError(
            f"JSON chat completion does not yet support provider type: {provider.type}"
        )
    if provider.base_url is None:
        raise JsonChatCompletionError("Provider base_url is required for chat completions")

    payload = {
        "model": model_ref.model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {resolve_provider_api_key(provider)}",
        "Content-Type": "application/json",
    }
    base_url = provider.base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=provider.timeout_s) as client:
        try:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise JsonChatCompletionError(
                f"Chat completion request failed: {type(exc).__name__}"
            ) from exc
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JsonChatCompletionError("Chat completion response missing message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise JsonChatCompletionError("Chat completion returned empty content")
    return {"content": content, "raw_response": data}
