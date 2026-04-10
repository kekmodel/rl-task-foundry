"""Tests for config loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rlvr_synth.config import load_config, RlvrSynthConfig


def test_load_minimal_config(tmp_path: Path) -> None:
    """Config loads from YAML and env vars are expanded."""
    yaml_content = textwrap.dedent("""\
        database:
          connection: "postgresql://u:p@localhost:5432/db"
          schema: "public"
          anchors:
            - table: "customers"
              column: "id"
          pk_sample_size: 10
          pk_sample_strategy: "random"
          pool_min_size: 2
          pool_max_size: 10
          max_inflight_queries: 8
          control_plane_pool_size: 2
          statement_timeout_ms: 3000
          connection_acquire_timeout_ms: 3000
          read_only: true
        domain:
          scenario: "test"
          user_role: "user"
          agent_role: "agent"
          language: "en"
        providers:
          default: "test_provider"
          test_provider:
            type: "openai_compatible"
            base_url: "http://localhost:8000/v1"
            api_key: "test"
            model: "test-model"
            max_concurrent: 4
            max_pending: 8
        models:
          tool_architect:
            provider: "test_provider"
            model: "test-model"
          task_composer:
            provider: "test_provider"
            model: "test-model"
          solver:
            - provider: "test_provider"
              count: 2
        provider_resilience:
          circuit_breaker_window_s: 30
          circuit_breaker_threshold: 0.5
          probe_interval_s: 10
          release_semaphore_on_backoff: true
          gpu_oom_action: "reduce_batch"
          queue_full_action: "backpressure"
          health_check_endpoint: "/health"
          health_check_interval_s: 5
        tool_architect:
          base_strategy: "one_per_table"
          hop_constraint: "single"
          variants:
            - level: 1
              naming: "direct"
              column_split: false
              decoy_scope: "all"
          validation:
            sample_per_tool: 2
            max_retry: 1
        schema_explorer:
          max_depth: 3
          max_paths: 50
          max_fanout: 20
          exclude_tables: []
          exclude_columns: []
        task_composer:
          initial_depth: 2
          initial_conditions: 1
          initial_tool_level: 1
          max_depth: 4
          max_conditions: 3
          negative_outcome_ratio: 0.1
          text_truncation_length: 100
        solver:
          num_solvers: 2
          max_turns: 10
          timeout: 30
        calibration:
          target_pass_rate: 0.5
          sigma: 0.5
          max_iterations: 5
          max_concurrent_composers: 2
          max_concurrent_solver_pks: 4
          ci_alpha: 0.1
          max_compose_retries: 2
          solver_early_termination: true
          difficulty_step:
            depth: 1
            conditions: 1
            tool_level: 1
        verification:
          mode: "exact_and_partial"
          partial_threshold: 0.8
          shadow_sample_rate: 0.1
          shadow_disagreement_threshold: 0.05
          label_tier: "A"
          tier_b_rules:
            float_precision: 2
            tie_breaker: "pk_asc"
            date_granularity: "date"
            null_handling: "exclude"
            list_ordering: "pk_asc"
        output:
          format: "jsonl"
          path: "./output/test.jsonl"
          include_metadata: true
          include_tool_traces: false
          include_rollout_records: false
          dedup_similarity_threshold: 0.85
        budget:
          mode: "hybrid"
          max_api_usd: 10.0
          max_gpu_hours: 0.5
          abort_if_accept_rate_below: 0.05
          reservation_mode: "phase_specific"
          smoke_canary_pks: 2
          smoke_canary_solvers: 1
          load_canary_pks: 4
          load_canary_min_completed: 2
        privacy:
          auto_detect_pii: true
          pii_patterns: ["email", "phone"]
          redact_dashboard: true
          redact_event_bus: true
        dashboard:
          enabled: false
          refresh_rate: 1.0
    """)
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content)

    cfg = load_config(str(config_path))

    assert isinstance(cfg, RlvrSynthConfig)
    assert cfg.database.pk_sample_size == 10
    assert cfg.database.anchors[0].table == "customers"
    assert cfg.domain.language == "en"
    assert cfg.providers.default == "test_provider"
    assert cfg.models.solver[0].count == 2
    assert cfg.calibration.max_concurrent_solver_pks == 4
    assert cfg.verification.label_tier == "A"
    assert cfg.budget.mode == "hybrid"
    assert cfg.privacy.auto_detect_pii is True


def test_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables in config values are expanded."""
    monkeypatch.setenv("TEST_DB_PASS", "secret123")
    yaml_content = textwrap.dedent("""\
        database:
          connection: "postgresql://u:${TEST_DB_PASS}@localhost/db"
          schema: "public"
          anchors:
            - table: "t"
              column: "id"
          pk_sample_size: 1
          pk_sample_strategy: "random"
          pool_min_size: 1
          pool_max_size: 5
          max_inflight_queries: 3
          control_plane_pool_size: 1
          statement_timeout_ms: 1000
          connection_acquire_timeout_ms: 1000
          read_only: true
        domain:
          scenario: "t"
          user_role: "u"
          agent_role: "a"
          language: "en"
        providers:
          default: "p"
          p:
            type: "openai_compatible"
            base_url: "http://localhost/v1"
            api_key: "k"
            model: "m"
            max_concurrent: 1
            max_pending: 1
        models:
          tool_architect:
            provider: "p"
            model: "m"
          task_composer:
            provider: "p"
            model: "m"
          solver:
            - provider: "p"
              count: 1
        provider_resilience:
          circuit_breaker_window_s: 10
          circuit_breaker_threshold: 0.5
          probe_interval_s: 5
          release_semaphore_on_backoff: true
          gpu_oom_action: "reduce_batch"
          queue_full_action: "backpressure"
          health_check_endpoint: "/health"
          health_check_interval_s: 5
        tool_architect:
          base_strategy: "one_per_table"
          hop_constraint: "single"
          variants: []
          validation:
            sample_per_tool: 1
            max_retry: 1
        schema_explorer:
          max_depth: 2
          max_paths: 10
          max_fanout: 10
          exclude_tables: []
          exclude_columns: []
        task_composer:
          initial_depth: 1
          initial_conditions: 1
          initial_tool_level: 1
          max_depth: 2
          max_conditions: 1
          negative_outcome_ratio: 0.0
          text_truncation_length: 50
        solver:
          num_solvers: 1
          max_turns: 5
          timeout: 10
        calibration:
          target_pass_rate: 0.5
          sigma: 0.5
          max_iterations: 3
          max_concurrent_composers: 1
          max_concurrent_solver_pks: 1
          ci_alpha: 0.1
          max_compose_retries: 1
          solver_early_termination: false
          difficulty_step:
            depth: 1
            conditions: 1
            tool_level: 1
        verification:
          mode: "exact"
          partial_threshold: 0.8
          shadow_sample_rate: 0.0
          shadow_disagreement_threshold: 0.05
          label_tier: "A"
          tier_b_rules:
            float_precision: 2
            tie_breaker: "pk_asc"
            date_granularity: "date"
            null_handling: "exclude"
            list_ordering: "pk_asc"
        output:
          format: "jsonl"
          path: "./out.jsonl"
          include_metadata: true
          include_tool_traces: false
          include_rollout_records: false
          dedup_similarity_threshold: 0.9
        budget:
          mode: "hybrid"
          max_api_usd: 1.0
          max_gpu_hours: 0.1
          abort_if_accept_rate_below: 0.01
          reservation_mode: "phase_specific"
          smoke_canary_pks: 1
          smoke_canary_solvers: 1
          load_canary_pks: 2
          load_canary_min_completed: 1
        privacy:
          auto_detect_pii: false
          pii_patterns: []
          redact_dashboard: false
          redact_event_bus: false
        dashboard:
          enabled: false
          refresh_rate: 1.0
    """)
    config_path = tmp_path / "env_config.yaml"
    config_path.write_text(yaml_content)

    cfg = load_config(str(config_path))
    assert "secret123" in cfg.database.connection
    assert "${TEST_DB_PASS}" not in cfg.database.connection
