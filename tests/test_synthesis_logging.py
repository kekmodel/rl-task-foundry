from __future__ import annotations

import json
from pathlib import Path

from rl_task_foundry.infra.event_log import TrialEventLogger
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger
from rl_task_foundry.synthesis.pipeline_events import PipelineFlowLogger


def test_phase_monitor_logger_reuses_file_handles_and_writes_mirror(tmp_path: Path) -> None:
    primary = tmp_path / "phase_monitors.jsonl"
    mirror = tmp_path / "mirror" / "phase_monitors.jsonl"
    logger = PipelinePhaseMonitorLogger(
        phase_monitor_log_path=primary,
        mirror_phase_monitor_log_path=mirror,
        flow_kind="trial",
        flow_id="flow-1",
    )

    logger.emit(phase="phase_a", status="started")
    first_primary_handle = logger._primary_sink._handle
    first_mirror_handle = logger._mirror_sink._handle if logger._mirror_sink is not None else None
    logger.emit(phase="phase_a", status="completed")
    second_primary_handle = logger._primary_sink._handle
    second_mirror_handle = logger._mirror_sink._handle if logger._mirror_sink is not None else None
    logger.close()

    assert first_primary_handle is not None
    assert first_primary_handle is second_primary_handle
    assert first_mirror_handle is not None
    assert first_mirror_handle is second_mirror_handle
    lines = [json.loads(line) for line in primary.read_text(encoding="utf-8").splitlines()]
    mirror_lines = [json.loads(line) for line in mirror.read_text(encoding="utf-8").splitlines()]
    assert [line["status"] for line in lines] == ["started", "completed"]
    assert mirror_lines == lines


def test_pipeline_flow_logger_reuses_file_handle(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = PipelineFlowLogger(
        event_log_path=path,
        flow_kind="flow",
        flow_id="flow-1",
    )

    logger.emit(stage="synthesis", status="started")
    first_handle = logger._primary_sink._handle
    logger.emit(stage="synthesis", status="completed")
    second_handle = logger._primary_sink._handle
    logger.close()

    assert first_handle is not None
    assert first_handle is second_handle
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["status"] for line in lines] == ["started", "completed"]


def test_trial_event_logger_flushes_sidecar_records_immediately(tmp_path: Path) -> None:
    logger = TrialEventLogger(tmp_path / "trial_events.jsonl")

    sidecar = logger.write_sidecar_jsonl(
        "reasoning_content.jsonl",
        [{"actor": "composer", "raw_item": {"type": "reasoning"}}],
    )

    assert sidecar == tmp_path / "reasoning_content.jsonl"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {
        "actor": "composer",
        "raw_item": {"type": "reasoning"},
    }
    logger.close()
