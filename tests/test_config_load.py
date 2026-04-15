from pathlib import Path

from rl_task_foundry.config import load_config


def test_load_config_uses_solver_run_count_source_of_truth():
    config = load_config(Path("rl_task_foundry.yaml"))
    assert config.models.total_solver_runs == 6
    assert config.output.run_db_path.name == "run.db"
    assert config.database.dsn == "postgresql://sakila:sakila@127.0.0.1:5433/sakila"
    assert config.budget.max_gpu_hours is None
    assert "codex_oauth" in config.providers
    assert "local_server" in config.providers
    assert config.providers["local_server"].base_url == "http://127.0.0.1:8000/v1"
    assert config.atomic_tools.max_tools == 300
    assert config.atomic_tools.bounded_result_limit == 100
    assert config.atomic_tools.max_batch_values == 128
    assert config.atomic_tools.float_precision == 2
    assert config.synthesis.runtime.max_turns == 20
    assert config.synthesis.runtime.max_generation_attempts == 5
    assert config.synthesis.runtime.max_consecutive_category_discards == 3
    assert config.synthesis.runtime.category_backoff_duration_s == 3600
    assert config.synthesis.runtime.schema_summary_max_tables == 32
    assert config.synthesis.runtime.tool_surface_summary_max_entries == 24
    assert config.synthesis.runtime.prompt_schema_orientation_max_tables == 8
    assert config.synthesis.runtime.prompt_schema_orientation_max_columns == 8
    assert config.synthesis.runtime.prompt_tool_surface_hint_limit == 16
    assert config.synthesis.runtime.label_preview_field_limit == 8
    assert config.synthesis.runtime.diagnostic_item_limit == 5
    assert config.synthesis.runtime.recent_tool_call_limit == 20
    assert config.synthesis.runtime.payload_preview_max_string_length == 400
    assert config.synthesis.runtime.payload_preview_max_list_items == 3
    assert config.synthesis.runtime.payload_preview_max_dict_items == 6
    assert config.synthesis.coverage_planner.target_count_per_band == 3
    assert config.calibration.solver_batch_size == 3
    assert config.calibration.max_solver_runs == 30
    assert config.provider_resilience.minimum_request_count == 2
    assert config.task_registry.minhash_num_perm == 128
    assert config.task_registry.default_query_limit == 20
    assert config.task_registry.semantic_shingle_size == 3
    assert config.budget.min_accept_rate_attempts == 10
    assert config.estimated_total_db_connections == 40


def test_load_config_applies_runtime_model_overrides():
    config = load_config(
        Path("rl_task_foundry.yaml"),
        composer_provider="local_server",
        solver_provider="local_server",
        solver_model="local-gpt",
    )
    assert config.models.composer.provider == "local_server"
    assert all(solver.provider == "local_server" for solver in config.models.solvers)
    assert all(solver.model == "local-gpt" for solver in config.models.solvers)
