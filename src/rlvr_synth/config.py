"""Config loading — YAML to typed Pydantic models with env var expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# ── Leaf models ──────────────────────────────────


class AnchorConfig(BaseModel):
    table: str
    column: str


class DatabaseConfig(BaseModel):
    connection: str
    schema_: str = Field(alias="schema")
    anchors: list[AnchorConfig]
    pk_sample_size: int
    pk_sample_strategy: Literal["random", "sequential", "filtered"]
    pk_filter: str | None = None
    pool_min_size: int
    pool_max_size: int
    max_inflight_queries: int
    control_plane_pool_size: int
    statement_timeout_ms: int
    connection_acquire_timeout_ms: int
    read_only: bool


class DomainConfig(BaseModel):
    scenario: str
    user_role: str
    agent_role: str
    language: str


class ProviderEntry(BaseModel):
    type: Literal["openai_compatible", "anthropic", "openai", "google"]
    base_url: str | None = None
    api_key: str
    model: str | None = None
    max_concurrent: int
    max_pending: int | None = None


class ProvidersConfig(BaseModel):
    default: str
    # Dynamic provider entries stored as extra fields
    model_config = {"extra": "allow"}

    def get_provider(self, name: str) -> ProviderEntry:
        raw = getattr(self, name, None)
        if raw is None:
            extras = self.__pydantic_extra__ or {}
            raw = extras.get(name)
        if raw is None:
            raise KeyError(f"Provider '{name}' not found")
        if isinstance(raw, dict):
            return ProviderEntry(**raw)
        return raw


class ModelRef(BaseModel):
    provider: str
    model: str


class SolverModelRef(BaseModel):
    provider: str
    count: int


class ModelsConfig(BaseModel):
    tool_architect: ModelRef
    task_composer: ModelRef
    solver: list[SolverModelRef]


class ProviderResilienceConfig(BaseModel):
    circuit_breaker_window_s: int
    circuit_breaker_threshold: float
    probe_interval_s: int
    release_semaphore_on_backoff: bool
    gpu_oom_action: Literal["reduce_batch", "stop"]
    queue_full_action: Literal["backpressure", "drop"]
    health_check_endpoint: str
    health_check_interval_s: int


class ToolVariantConfig(BaseModel):
    level: int
    naming: Literal["direct", "semi_indirect", "business_domain"]
    column_split: bool
    decoy_scope: Literal["all", "related", "none"]


class ToolValidationConfig(BaseModel):
    sample_per_tool: int
    max_retry: int


class ToolArchitectConfig(BaseModel):
    base_strategy: Literal["one_per_table", "one_per_column_group"]
    hop_constraint: Literal["single", "direct"]
    variants: list[ToolVariantConfig]
    validation: ToolValidationConfig


class SchemaExplorerConfig(BaseModel):
    max_depth: int
    max_paths: int
    max_fanout: int
    exclude_tables: list[str]
    exclude_columns: list[str]


class TaskComposerConfig(BaseModel):
    initial_depth: int
    initial_conditions: int
    initial_tool_level: int
    max_depth: int
    max_conditions: int
    negative_outcome_ratio: float
    text_truncation_length: int


class SolverConfig(BaseModel):
    num_solvers: int
    max_turns: int
    timeout: int


class DifficultyStepConfig(BaseModel):
    depth: int
    conditions: int
    tool_level: int


class CalibrationConfig(BaseModel):
    target_pass_rate: float
    sigma: float
    max_iterations: int
    max_concurrent_composers: int
    max_concurrent_solver_pks: int
    ci_alpha: float
    max_compose_retries: int
    solver_early_termination: bool
    difficulty_step: DifficultyStepConfig


class TierBRulesConfig(BaseModel):
    float_precision: int
    tie_breaker: str
    date_granularity: str
    null_handling: str
    list_ordering: str


class VerificationConfig(BaseModel):
    mode: Literal["exact", "partial", "exact_and_partial"]
    partial_threshold: float
    shadow_sample_rate: float
    shadow_disagreement_threshold: float
    label_tier: Literal["A", "A+B"]
    tier_b_rules: TierBRulesConfig


class OutputConfig(BaseModel):
    format: Literal["jsonl"]
    path: str
    include_metadata: bool
    include_tool_traces: bool
    include_rollout_records: bool
    dedup_similarity_threshold: float


class BudgetConfig(BaseModel):
    mode: Literal["hybrid", "token_cost", "gpu_hours"]
    max_api_usd: float
    max_gpu_hours: float
    abort_if_accept_rate_below: float
    reservation_mode: Literal["phase_specific", "flat"]
    smoke_canary_pks: int
    smoke_canary_solvers: int
    load_canary_pks: int
    load_canary_min_completed: int


class PrivacyConfig(BaseModel):
    auto_detect_pii: bool
    pii_patterns: list[str]
    redact_dashboard: bool
    redact_event_bus: bool


class DashboardConfig(BaseModel):
    enabled: bool
    refresh_rate: float


# ── Root model ───────────────────────────────────


class RlvrSynthConfig(BaseModel):
    database: DatabaseConfig
    domain: DomainConfig
    providers: ProvidersConfig
    models: ModelsConfig
    provider_resilience: ProviderResilienceConfig
    tool_architect: ToolArchitectConfig
    schema_explorer: SchemaExplorerConfig
    task_composer: TaskComposerConfig
    solver: SolverConfig
    calibration: CalibrationConfig
    verification: VerificationConfig
    output: OutputConfig
    budget: BudgetConfig
    privacy: PrivacyConfig
    dashboard: DashboardConfig


# ── Loader ───────────────────────────────────────

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(obj: object) -> object:
    """Recursively expand ${VAR} patterns in string values."""
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_config(path: str | Path) -> RlvrSynthConfig:
    """Load and validate config from a YAML file."""
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    data = _expand_env_vars(data)
    return RlvrSynthConfig(**data)
