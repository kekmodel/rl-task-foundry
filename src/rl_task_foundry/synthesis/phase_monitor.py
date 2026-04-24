"""Contract-oriented phase monitor logging for synthesis pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rl_task_foundry.synthesis.jsonl_logger import JsonlFileSink


def default_phase_monitor_log_path(traces_dir: Path) -> Path:
    """Store phase monitors next to the trace root, not inside it."""

    return traces_dir.parent / "phase_monitors.jsonl"


@dataclass(frozen=True, slots=True)
class PipelinePhaseMonitorRecord:
    flow_kind: str
    flow_id: str
    seq: int
    timestamp: str
    phase: str
    status: str
    expected_contract: dict[str, object]
    actual_data: dict[str, object]
    checks: dict[str, object]
    diagnostics: dict[str, object]


@dataclass(slots=True)
class PipelinePhaseMonitorLogger:
    phase_monitor_log_path: Path
    flow_kind: str
    flow_id: str
    mirror_phase_monitor_log_path: Path | None = None
    event_logger: object | None = None
    _seq: int = field(default=0, init=False, repr=False)
    _primary_sink: JsonlFileSink = field(init=False, repr=False)
    _mirror_sink: JsonlFileSink | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._primary_sink = JsonlFileSink(self.phase_monitor_log_path)
        if (
            self.mirror_phase_monitor_log_path is not None
            and self.mirror_phase_monitor_log_path != self.phase_monitor_log_path
        ):
            self._mirror_sink = JsonlFileSink(self.mirror_phase_monitor_log_path)

    def emit(
        self,
        *,
        phase: str,
        status: str,
        expected_contract: dict[str, object] | None = None,
        actual_data: dict[str, object] | None = None,
        checks: dict[str, object] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> PipelinePhaseMonitorRecord:
        self._seq += 1
        record = PipelinePhaseMonitorRecord(
            flow_kind=self.flow_kind,
            flow_id=self.flow_id,
            seq=self._seq,
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            status=status,
            expected_contract=dict(expected_contract or {}),
            actual_data=dict(actual_data or {}),
            checks=dict(checks or {}),
            diagnostics=dict(diagnostics or {}),
        )
        payload = {
            "flow_kind": record.flow_kind,
            "flow_id": record.flow_id,
            "seq": record.seq,
            "timestamp": record.timestamp,
            "phase": record.phase,
            "status": record.status,
            "expected_contract": record.expected_contract,
            "actual_data": record.actual_data,
            "checks": record.checks,
            "diagnostics": record.diagnostics,
        }
        self._primary_sink.write_record(payload)
        if self._mirror_sink is not None:
            self._mirror_sink.write_record(payload)
        if self.event_logger is not None:
            self.event_logger.log_sync(
                actor="phase",
                event_type=f"{record.phase}.{record.status}",
                payload={
                    "seq": record.seq,
                    "expected_contract": record.expected_contract,
                    "actual_data": record.actual_data,
                    "checks": record.checks,
                    "diagnostics": record.diagnostics,
                },
            )
        return record

    def close(self) -> None:
        self._primary_sink.close()
        if self._mirror_sink is not None:
            self._mirror_sink.close()
