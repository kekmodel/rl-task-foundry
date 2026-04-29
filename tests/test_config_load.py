from pathlib import Path

from rl_task_foundry.config import load_config


def _write_config_from_repo_default(
    source: Path,
    target: Path,
    *,
    dsn_expr: str,
) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        '  dsn: "${DATABASE_DSN:-postgresql://pagila:pagila@127.0.0.1:5433/pagila}"',
        f'  dsn: "{dsn_expr}"',
        1,
    )
    target.write_text(text, encoding="utf-8")


def test_load_config_uses_solver_run_count_source_of_truth():
    config = load_config(Path("rl_task_foundry.yaml"))
    assert config.models.solver_model_count == 1
    assert config.output.run_db_path.name == "run.db"
    assert config.database.dsn == "postgresql://pagila:pagila@127.0.0.1:5433/pagila"
    assert config.budget.max_gpu_hours is None
    assert config.models.composer.provider == "openrouter"
    assert config.models.composer.model == "moonshotai/kimi-k2.5"
    assert {solver.provider for solver in config.models.solvers} == {"openrouter"}
    assert {solver.model for solver in config.models.solvers} == {
        "moonshotai/kimi-k2.5"
    }
    assert "openai_api" in config.providers
    assert config.providers["openai_api"].type == "openai"
    assert config.providers["openai_api"].api_key_env == "OPENAI_API_KEY"
    assert config.providers["openrouter"].base_url == "https://openrouter.ai/api/v1"
    assert config.providers["openrouter"].api_key_env == "OPENROUTER_API_KEY"
    assert config.providers["openrouter"].timeout_s == 180
    assert config.providers["openrouter"].max_retries == 3
    assert "codex_oauth" in config.providers
    assert "local_server" in config.providers
    assert config.providers["local_server"].base_url == "http://127.0.0.1:8000/v1"
    assert config.atomic_tools.max_tools == 300
    assert config.atomic_tools.bounded_result_limit == 100
    assert config.atomic_tools.max_batch_values == 128
    assert config.atomic_tools.float_precision == 2
    assert config.synthesis.runtime.max_turns == 30
    assert config.synthesis.runtime.trial_timeout_s == 300
    assert config.synthesis.runtime.run_timeout_s == 300
    assert config.synthesis.runtime.anchor_candidates_enabled is True
    assert config.synthesis.runtime.anchor_candidate_limit == 10
    assert config.synthesis.runtime.max_generation_attempts == 5
    assert config.synthesis.runtime.max_consecutive_category_discards == 3
    assert config.synthesis.runtime.category_backoff_duration_s == 3600
    assert config.synthesis.runtime.schema_summary_max_tables == 32
    assert config.synthesis.runtime.prompt_schema_orientation_max_tables == 8
    assert config.synthesis.runtime.prompt_schema_orientation_max_columns == 8
    assert config.synthesis.runtime.label_preview_field_limit == 8
    assert config.synthesis.runtime.diagnostic_item_limit == 5
    assert config.synthesis.runtime.recent_tool_call_limit == 20
    assert config.synthesis.runtime.payload_preview_max_string_length == 400
    assert config.synthesis.runtime.payload_preview_max_list_items == 3
    assert config.synthesis.runtime.payload_preview_max_dict_items == 6
    assert config.synthesis.coverage_planner.target_count_per_band == 3
    assert config.solver_runtime.max_episode_duration_ms == 120000
    assert config.calibration.lower_pass_rate == 0.5
    assert config.calibration.upper_pass_rate == 0.9
    assert config.calibration.ci_alpha == 0.1
    assert config.calibration.solver_batch_size == 4
    assert config.calibration.max_solver_runs == 20
    assert config.provider_resilience.minimum_request_count == 2
    assert config.task_registry.minhash_num_perm == 128
    assert config.task_registry.default_query_limit == 20
    assert config.task_registry.semantic_shingle_size == 3
    assert config.budget.min_accept_rate_attempts == 10
    assert config.visibility.default_visibility == "user_visible"
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


def test_load_config_reads_cwd_dotenv_for_tmp_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_config = Path.cwd() / "rl_task_foundry.yaml"
    config_dir = tmp_path / "tmp_configs"
    config_dir.mkdir()
    config_path = config_dir / "trial.yaml"
    _write_config_from_repo_default(
        repo_config,
        config_path,
        dsn_expr="${TMP_TRIAL_DSN:-postgresql://fallback/fallback}",
    )
    (tmp_path / ".env").write_text(
        "TMP_TRIAL_DSN=postgresql://root-env/root\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TMP_TRIAL_DSN", raising=False)
    monkeypatch.chdir(tmp_path)

    config = load_config(config_path)

    assert config.database.dsn == "postgresql://root-env/root"


def test_load_config_prefers_config_local_dotenv_over_cwd_dotenv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_config = Path.cwd() / "rl_task_foundry.yaml"
    config_dir = tmp_path / "tmp_configs"
    config_dir.mkdir()
    config_path = config_dir / "trial.yaml"
    _write_config_from_repo_default(
        repo_config,
        config_path,
        dsn_expr="${TMP_TRIAL_DSN:-postgresql://fallback/fallback}",
    )
    (tmp_path / ".env").write_text(
        "TMP_TRIAL_DSN=postgresql://root-env/root\n",
        encoding="utf-8",
    )
    (config_dir / ".env").write_text(
        "TMP_TRIAL_DSN=postgresql://config-env/config\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TMP_TRIAL_DSN", raising=False)
    monkeypatch.chdir(tmp_path)

    config = load_config(config_path)

    assert config.database.dsn == "postgresql://config-env/config"
