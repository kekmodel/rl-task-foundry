from __future__ import annotations

from pathlib import Path

from rl_task_foundry.config.models import ModelRef, ProviderConfig, SynthesisRuntimeConfig
from rl_task_foundry.synthesis.backend_openai_agents import OpenAIAgentsSynthesisBackend


def test_synthesis_backend_write_artifact_creates_dir_once_per_kind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(),
        traces_dir=tmp_path / "traces",
    )

    mkdir_calls: list[Path] = []
    original_mkdir = Path.mkdir

    def _recording_mkdir(self: Path, *args, **kwargs):
        mkdir_calls.append(self)
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _recording_mkdir)

    backend._write_artifact(
        kind="transcripts",
        db_id="sakila",
        requested_topic="assignment",
        payload={"a": 1},
    )
    first_transcript_dir_calls = mkdir_calls.count(tmp_path / "traces" / "transcripts")
    backend._write_artifact(
        kind="transcripts",
        db_id="sakila",
        requested_topic="assignment",
        payload={"a": 2},
    )

    transcript_dir = tmp_path / "traces" / "transcripts"
    assert mkdir_calls.count(transcript_dir) == first_transcript_dir_calls
