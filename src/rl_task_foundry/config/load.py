"""Configuration loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from rl_task_foundry.config.models import AppConfig, ModelsConfig, SolverModelConfig

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(env_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            return match.group(0)

        return _ENV_PATTERN.sub(replace, value)
    return value


def _validate_provider_name(config: AppConfig, provider_name: str) -> None:
    if provider_name not in config.providers:
        raise ValueError(f"Unknown provider override: {provider_name}")


def apply_model_overrides(
    config: AppConfig,
    *,
    composer_provider: str | None = None,
    composer_model: str | None = None,
    solver_provider: str | None = None,
    solver_model: str | None = None,
) -> AppConfig:
    """Apply runtime model/provider overrides without mutating the source config."""

    for provider_name in (composer_provider, solver_provider):
        if provider_name is not None:
            _validate_provider_name(config, provider_name)

    models = config.models.model_copy(deep=True)

    if composer_provider is not None:
        models.composer.provider = composer_provider
    if composer_model is not None:
        models.composer.model = composer_model

    if solver_provider is not None or solver_model is not None:
        updated_solvers: list[SolverModelConfig] = []
        for solver in models.solvers:
            updated = solver.model_copy(
                update={
                    "provider": solver_provider if solver_provider is not None else solver.provider,
                    "model": solver_model if solver_model is not None else solver.model,
                }
            )
            updated_solvers.append(updated)
        models = ModelsConfig(
            composer=models.composer,
            solvers=updated_solvers,
        )

    return config.model_copy(update={"models": models})


def load_config(
    path: str | Path,
    *,
    composer_provider: str | None = None,
    composer_model: str | None = None,
    solver_provider: str | None = None,
    solver_model: str | None = None,
) -> AppConfig:
    """Load and validate application configuration."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = AppConfig.model_validate(_expand_env(data))
    return apply_model_overrides(
        config,
        composer_provider=composer_provider,
        composer_model=composer_model,
        solver_provider=solver_provider,
        solver_model=solver_model,
    )
