"""Run manifest helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from rl_task_foundry.config.models import AppConfig


def config_hash(config: AppConfig) -> str:
    """Return a stable hash of the validated config."""

    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class RunManifest:
    run_id: str
    config_hash: str
    total_solver_replicas: int
