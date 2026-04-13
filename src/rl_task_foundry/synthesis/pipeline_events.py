"""Ordered JSONL event logging for synthesis pipeline flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from rl_task_foundry.synthesis.jsonl_logger import JsonlFileSink


def build_flow_id(flow_kind: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{flow_kind}:{timestamp}:{uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class PipelineFlowEvent:
    flow_kind: str
    flow_id: str
    seq: int
    timestamp: str
    stage: str
    status: str
    payload: dict[str, Any]


@dataclass(slots=True)
class PipelineFlowLogger:
    event_log_path: Path
    flow_kind: str
    flow_id: str
    mirror_event_log_path: Path | None = None
    _seq: int = field(default=0, init=False, repr=False)
    _primary_sink: JsonlFileSink = field(init=False, repr=False)
    _mirror_sink: JsonlFileSink | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._primary_sink = JsonlFileSink(self.event_log_path)
        if (
            self.mirror_event_log_path is not None
            and self.mirror_event_log_path != self.event_log_path
        ):
            self._mirror_sink = JsonlFileSink(self.mirror_event_log_path)

    def emit(
        self,
        *,
        stage: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> PipelineFlowEvent:
        self._seq += 1
        event = PipelineFlowEvent(
            flow_kind=self.flow_kind,
            flow_id=self.flow_id,
            seq=self._seq,
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            status=status,
            payload=dict(payload or {}),
        )
        payload_dict = {
            "flow_kind": event.flow_kind,
            "flow_id": event.flow_id,
            "seq": event.seq,
            "timestamp": event.timestamp,
            "stage": event.stage,
            "status": event.status,
            "payload": event.payload,
        }
        self._primary_sink.write_record(payload_dict)
        if self._mirror_sink is not None:
            self._mirror_sink.write_record(payload_dict)
        return event

    def close(self) -> None:
        self._primary_sink.close()
        if self._mirror_sink is not None:
            self._mirror_sink.close()
