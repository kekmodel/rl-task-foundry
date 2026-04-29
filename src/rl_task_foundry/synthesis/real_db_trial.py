"""Real-database single-task trial runner.

The on-disk source of truth for trial debugging is the phase monitor log plus
debug traces and any exported bundle artifacts. The summary object returned by
this module is in-memory only and is not persisted as a separate JSON file.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.db import DatabasePools
from rl_task_foundry.infra.event_log import TrialEventLogger
from rl_task_foundry.pipeline.solver_orchestrator import SolverOrchestrator
from rl_task_foundry.synthesis.bundle_exporter import TaskBundleExporter
from rl_task_foundry.synthesis.contracts import normalize_topic
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisArtifactGenerationError,
    SynthesisBackendFailure,
    SynthesisGenerationAttempt,
    SynthesisPhaseExecutionError,
    SynthesisProviderUnavailableError,
    SynthesisRuntimeError,
    _SynthesisBackendProtocol,
)
from rl_task_foundry.synthesis.snapshot_materializer import (
    SchemaSnapshotMaterializer,
)
from rl_task_foundry.synthesis.synthesis_db import SynthesisDb
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryCommitStatus,
    TaskRegistryWriter,
)


class RealDbTrialStatus(StrEnum):
    ACCEPTED = "accepted"
    REGISTRY_DUPLICATE = "registry_duplicate"
    SYNTHESIS_FAILED = "synthesis_failed"


@dataclass(init=False, frozen=True, slots=True)
class RealDbTrialSummary:
    db_id: str
    requested_topic: str
    trial_status: RealDbTrialStatus
    flow_id: str | None = None
    phase_monitor_log_path: Path | None = None
    debug_root: Path | None = None
    debug_traces_dir: Path | None = None
    synthesis_traces_dir: Path | None = None
    solver_traces_dir: Path | None = None
    task_id: str | None = None
    quality_gate_status: str | None = None
    synthesis_error_type: str | None = None
    synthesis_error_message: str | None = None
    synthesis_phase: str | None = None
    backend_failures: tuple[str, ...] = ()
    attempt_outcomes: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    solver_pass_rate: float | None = None
    solver_ci_low: float | None = None
    solver_ci_high: float | None = None
    solver_matched_runs: int | None = None
    solver_planned_runs: int | None = None
    solver_completed_runs: int | None = None
    solver_evaluable_runs: int | None = None
    solver_failed_runs: int | None = None
    feedback_events: int = 0
    last_feedback_error_codes: tuple[str, ...] = ()
    registry_status: TaskRegistryCommitStatus | None = None
    registry_task_id: str | None = None
    bundle_root: Path | None = None
    elapsed_seconds: float | None = None

    def __init__(
        self,
        *,
        db_id: str,
        requested_topic: str | None = None,
        trial_status: RealDbTrialStatus,
        flow_id: str | None = None,
        phase_monitor_log_path: Path | None = None,
        debug_root: Path | None = None,
        debug_traces_dir: Path | None = None,
        synthesis_traces_dir: Path | None = None,
        solver_traces_dir: Path | None = None,
        task_id: str | None = None,
        quality_gate_status: str | None = None,
        synthesis_error_type: str | None = None,
        synthesis_error_message: str | None = None,
        synthesis_phase: str | None = None,
        backend_failures: tuple[str, ...] = (),
        attempt_outcomes: tuple[str, ...] = (),
        error_codes: tuple[str, ...] = (),
        solver_pass_rate: float | None = None,
        solver_ci_low: float | None = None,
        solver_ci_high: float | None = None,
        solver_matched_runs: int | None = None,
        solver_planned_runs: int | None = None,
        solver_completed_runs: int | None = None,
        solver_evaluable_runs: int | None = None,
        solver_failed_runs: int | None = None,
        feedback_events: int = 0,
        last_feedback_error_codes: tuple[str, ...] = (),
        registry_status: TaskRegistryCommitStatus | None = None,
        registry_task_id: str | None = None,
        bundle_root: Path | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        object.__setattr__(self, "db_id", db_id)
        object.__setattr__(
            self,
            "requested_topic",
            normalize_topic(requested_topic) if requested_topic else "",
        )
        object.__setattr__(self, "trial_status", trial_status)
        object.__setattr__(self, "flow_id", flow_id)
        object.__setattr__(self, "phase_monitor_log_path", phase_monitor_log_path)
        object.__setattr__(self, "debug_root", debug_root)
        object.__setattr__(self, "debug_traces_dir", debug_traces_dir)
        object.__setattr__(self, "synthesis_traces_dir", synthesis_traces_dir)
        object.__setattr__(self, "solver_traces_dir", solver_traces_dir)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "quality_gate_status", quality_gate_status)
        object.__setattr__(self, "synthesis_error_type", synthesis_error_type)
        object.__setattr__(self, "synthesis_error_message", synthesis_error_message)
        object.__setattr__(self, "synthesis_phase", synthesis_phase)
        object.__setattr__(self, "backend_failures", backend_failures)
        object.__setattr__(self, "attempt_outcomes", attempt_outcomes)
        object.__setattr__(self, "error_codes", error_codes)
        object.__setattr__(self, "solver_pass_rate", solver_pass_rate)
        object.__setattr__(self, "solver_ci_low", solver_ci_low)
        object.__setattr__(self, "solver_ci_high", solver_ci_high)
        object.__setattr__(self, "solver_matched_runs", solver_matched_runs)
        object.__setattr__(self, "solver_planned_runs", solver_planned_runs)
        object.__setattr__(self, "solver_completed_runs", solver_completed_runs)
        object.__setattr__(self, "solver_evaluable_runs", solver_evaluable_runs)
        object.__setattr__(self, "solver_failed_runs", solver_failed_runs)
        object.__setattr__(self, "feedback_events", feedback_events)
        object.__setattr__(
            self, "last_feedback_error_codes", last_feedback_error_codes
        )
        object.__setattr__(self, "registry_status", registry_status)
        object.__setattr__(self, "registry_task_id", registry_task_id)
        object.__setattr__(self, "bundle_root", bundle_root)
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)


def _solver_summary_from_attempt(
    attempt: SynthesisGenerationAttempt | None,
) -> dict[str, object]:
    if attempt is None:
        return {}
    return {
        "solver_pass_rate": attempt.solver_pass_rate,
        "solver_ci_low": attempt.solver_ci_low,
        "solver_ci_high": attempt.solver_ci_high,
        "solver_matched_runs": attempt.solver_matched_runs,
        "solver_planned_runs": attempt.solver_planned_runs,
        "solver_completed_runs": attempt.solver_completed_runs,
        "solver_evaluable_runs": attempt.solver_evaluable_runs,
        "solver_failed_runs": attempt.solver_failed_runs,
    }


@dataclass(slots=True)
class RealDbTrialRunner:
    config: AppConfig
    synthesis_runtime: SynthesisAgentRuntime | None = None
    registry: TaskRegistryWriter | None = None
    exporter: TaskBundleExporter | None = None
    database_pools: DatabasePools | None = None
    solver_orchestrator: SolverOrchestrator | None = None
    synthesis_db: SynthesisDb | None = None
    synthesis_backends: list[_SynthesisBackendProtocol] | None = None
    export_trial_bundle: bool = True
    _owns_solver_orchestrator: bool = field(default=False, init=False, repr=False)
    _owns_registry: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = TaskRegistryWriter.for_config(self.config)
            self._owns_registry = True
        if self.exporter is None:
            assert self.registry.snapshot_materializer is not None
            self.exporter = TaskBundleExporter(
                registry=self.registry,
                snapshot_materializer=self.registry.snapshot_materializer,
            )
        if self.solver_orchestrator is None and self.synthesis_runtime is None:
            self.solver_orchestrator = SolverOrchestrator(
                self.config,
                database_pools=self.database_pools,
            )
            self._owns_solver_orchestrator = True

    async def run(
        self,
        output_root: Path,
        *,
        db_id: str,
        topic: str | None = None,
        mirror_monitor_path: Path | None = None,
    ) -> RealDbTrialSummary:
        if topic:
            topic = normalize_topic(topic)
        output_root.mkdir(parents=True, exist_ok=True)
        debug_root = output_root / "debug"
        debug_traces_dir = debug_root / "traces"
        phase_monitor_log_path = debug_root / "phase_monitors.jsonl"
        synthesis_traces_dir = debug_traces_dir / "synthesis"
        solver_traces_dir = debug_traces_dir
        synthesis_traces_dir.mkdir(parents=True, exist_ok=True)
        event_logger = TrialEventLogger(debug_root / "trial_events.jsonl")
        flow_id = build_flow_id("real_db_trial")
        event_logger.log_sync(
            actor="runner",
            event_type="trial_started",
            payload={
                "flow_id": flow_id,
                "db_id": db_id,
                "topic_experiment_hint": topic,
                "output_root": str(output_root),
            },
        )
        phase_monitor = PipelinePhaseMonitorLogger(
            phase_monitor_log_path=phase_monitor_log_path,
            flow_kind="real_db_trial",
            flow_id=flow_id,
            mirror_phase_monitor_log_path=mirror_monitor_path,
            event_logger=event_logger,
        )
        phase_monitor.emit(
            phase="trial",
            status="started",
            expected_contract={"db_id": db_id, "topic_experiment_hint": topic},
            actual_data={"output_root": output_root},
            checks={"debug_root_ready": debug_root.exists()},
            diagnostics={},
        )
        if self.export_trial_bundle:
            trial_materializer = SchemaSnapshotMaterializer(
                root_dir=debug_root / "databases"
            )
            if self.synthesis_db is not None:
                self.synthesis_db.use_snapshot_materializer(trial_materializer)
            if hasattr(self.exporter, "snapshot_materializer"):
                self.exporter = replace(
                    self.exporter, snapshot_materializer=trial_materializer
                )
        runtime = self._synthesis_runtime_for_trial(
            debug_traces_dir, phase_monitor, event_logger=event_logger
        )
        # The solver orchestrator was built in __post_init__ with the
        # original config.output.traces_dir (global). Redirect its
        # output to this trial's debug_traces_dir so solver transcripts
        # and tool traces land under artifacts/<trial>/debug/traces/.
        if self.solver_orchestrator is not None:
            self.solver_orchestrator.traces_dir_override = debug_traces_dir
            self.solver_orchestrator.event_logger = event_logger
        # Synchronize the bundle exporter's snapshot source with where
        # SynthesisDb actually materializes the schema during this
        # trial. Without this the exporter reads the global
        # ``artifacts/databases/<db_id>/`` path and raises
        # FileNotFoundError even though the snapshot was written into
        # ``<output_root>/debug/databases/<db_id>/``.
        production_loop_started_at = time.monotonic()

        def elapsed_seconds() -> float:
            return round(time.monotonic() - production_loop_started_at, 3)

        try:
            draft = await asyncio.wait_for(
                runtime.synthesize_environment_draft(
                    db_id=db_id,
                    requested_topic=topic,
                ),
                timeout=float(self.config.synthesis.runtime.trial_timeout_s),
            )
        except asyncio.TimeoutError:
            elapsed = elapsed_seconds()
            phase_monitor.emit(
                phase="trial",
                status="failed",
                expected_contract={
                    "trial_timeout_s": self.config.synthesis.runtime.trial_timeout_s
                },
                actual_data={
                    "db_id": db_id,
                    "topic_experiment_hint": topic,
                    "elapsed_seconds": elapsed,
                },
                checks={"within_trial_timeout": False},
                diagnostics={
                    "error_type": "TrialWallClockTimeout",
                    "error_message": (
                        "real-db trial exceeded trial_timeout_s="
                        f"{self.config.synthesis.runtime.trial_timeout_s}"
                    ),
                },
            )
            summary = RealDbTrialSummary(
                db_id=db_id,
                requested_topic=topic,
                trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                flow_id=flow_id,
                phase_monitor_log_path=phase_monitor_log_path,
                debug_root=debug_root,
                debug_traces_dir=debug_traces_dir,
                synthesis_traces_dir=synthesis_traces_dir,
                solver_traces_dir=solver_traces_dir,
                synthesis_error_type="TrialWallClockTimeout",
                synthesis_error_message=(
                    "real-db trial exceeded trial_timeout_s="
                    f"{self.config.synthesis.runtime.trial_timeout_s}"
                ),
                synthesis_phase="trial_timeout",
                error_codes=("trial_wall_clock_timeout",),
                elapsed_seconds=elapsed,
            )
            phase_monitor.close()
            event_logger.close()
            return summary
        except SynthesisArtifactGenerationError as exc:
            last_attempt = exc.attempts[-1] if exc.attempts else None
            diagnostics = exc.last_artifact_diagnostics
            error_codes = tuple(diagnostics.error_codes if diagnostics else ())
            last_feedback_error_codes = tuple(
                diagnostics.last_feedback_error_codes if diagnostics else ()
            )
            feedback_events = diagnostics.feedback_events if diagnostics else 0
            solver_summary = _solver_summary_from_attempt(last_attempt)
            phase_monitor.emit(
                phase="synthesis",
                status="failed",
                expected_contract={"accepted_draft_required": True},
                actual_data={"db_id": db_id, "topic_experiment_hint": topic},
                checks={"accepted_draft_present": False},
                diagnostics={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "attempt_outcomes": [attempt.outcome.value for attempt in exc.attempts],
                    "error_codes": list(error_codes),
                    "feedback_events": feedback_events,
                    "last_feedback_error_codes": list(last_feedback_error_codes),
                    **solver_summary,
                },
            )
            summary = RealDbTrialSummary(
                db_id=db_id,
                requested_topic=topic,
                trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                flow_id=flow_id,
                phase_monitor_log_path=phase_monitor_log_path,
                debug_root=debug_root,
                debug_traces_dir=debug_traces_dir,
                synthesis_traces_dir=synthesis_traces_dir,
                solver_traces_dir=solver_traces_dir,
                synthesis_error_type=type(exc).__name__,
                synthesis_error_message=str(exc),
                attempt_outcomes=tuple(attempt.outcome.value for attempt in exc.attempts),
                error_codes=error_codes,
                feedback_events=feedback_events,
                last_feedback_error_codes=last_feedback_error_codes,
                elapsed_seconds=elapsed_seconds(),
                **solver_summary,
            )
            phase_monitor.close()
            event_logger.close()
            return summary
        except SynthesisPhaseExecutionError as exc:
            phase_monitor.emit(
                phase="synthesis",
                status="failed",
                expected_contract={"accepted_draft_required": True},
                actual_data={"db_id": db_id, "topic_experiment_hint": topic},
                checks={"accepted_draft_present": False},
                diagnostics={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "phase": exc.phase,
                    "backend_failures": list(_encode_backend_failures(exc.backend_failures)),
                },
            )
            summary = RealDbTrialSummary(
                db_id=db_id,
                requested_topic=topic,
                trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                flow_id=flow_id,
                phase_monitor_log_path=phase_monitor_log_path,
                debug_root=debug_root,
                debug_traces_dir=debug_traces_dir,
                synthesis_traces_dir=synthesis_traces_dir,
                solver_traces_dir=solver_traces_dir,
                synthesis_error_type=type(exc).__name__,
                synthesis_error_message=str(exc),
                synthesis_phase=exc.phase,
                backend_failures=_encode_backend_failures(exc.backend_failures),
                elapsed_seconds=elapsed_seconds(),
            )
            phase_monitor.close()
            event_logger.close()
            return summary
        except SynthesisProviderUnavailableError as exc:
            phase_monitor.emit(
                phase="synthesis",
                status="failed",
                expected_contract={"accepted_draft_required": True},
                actual_data={"db_id": db_id, "topic_experiment_hint": topic},
                checks={"accepted_draft_present": False},
                diagnostics={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "phase": exc.phase,
                },
            )
            summary = RealDbTrialSummary(
                db_id=db_id,
                requested_topic=topic,
                trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                flow_id=flow_id,
                phase_monitor_log_path=phase_monitor_log_path,
                debug_root=debug_root,
                debug_traces_dir=debug_traces_dir,
                synthesis_traces_dir=synthesis_traces_dir,
                solver_traces_dir=solver_traces_dir,
                synthesis_error_type=type(exc).__name__,
                synthesis_error_message=str(exc),
                synthesis_phase=exc.phase,
                elapsed_seconds=elapsed_seconds(),
            )
            phase_monitor.close()
            event_logger.close()
            return summary
        except SynthesisRuntimeError as exc:
            phase_monitor.emit(
                phase="synthesis",
                status="failed",
                expected_contract={"accepted_draft_required": True},
                actual_data={"db_id": db_id, "topic_experiment_hint": topic},
                checks={"accepted_draft_present": False},
                diagnostics={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            summary = RealDbTrialSummary(
                db_id=db_id,
                requested_topic=topic,
                trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                flow_id=flow_id,
                phase_monitor_log_path=phase_monitor_log_path,
                debug_root=debug_root,
                debug_traces_dir=debug_traces_dir,
                synthesis_traces_dir=synthesis_traces_dir,
                solver_traces_dir=solver_traces_dir,
                synthesis_error_type=type(exc).__name__,
                synthesis_error_message=str(exc),
                elapsed_seconds=elapsed_seconds(),
            )
            phase_monitor.close()
            event_logger.close()
            return summary

        assert self.registry is not None
        assert self.exporter is not None
        commit_result = self.registry.commit_draft(draft)
        phase_monitor.emit(
            phase="registry_commit",
            status=commit_result.status.value,
            expected_contract={"accepted_draft_required": True},
            actual_data={
                "task_id": draft.task_bundle.task_id,
                "registry_task_id": commit_result.task_id,
            },
            checks={
                "committed_or_duplicate": commit_result.status
                in {
                    TaskRegistryCommitStatus.COMMITTED,
                    TaskRegistryCommitStatus.DUPLICATE,
                }
            },
            diagnostics={},
        )
        bundle_root = output_root / "bundle" if self.export_trial_bundle else None
        if self.export_trial_bundle:
            assert bundle_root is not None
            self.exporter.export_bundle(bundle_root, task_id=commit_result.task_id)
            phase_monitor.emit(
                phase="bundle_export",
                status="completed",
                expected_contract={"bundle_root": bundle_root},
                actual_data={
                    "bundle_root": bundle_root,
                    "task_id": commit_result.task_id,
                },
                checks={"bundle_root_exists": bundle_root.exists()},
                diagnostics={},
            )
        else:
            phase_monitor.emit(
                phase="bundle_export",
                status="skipped",
                expected_contract={"registry_is_durable_source": True},
                actual_data={"task_id": commit_result.task_id},
                checks={"trial_bundle_export_disabled": True},
                diagnostics={
                    "export_hint": (
                        "Use export-bundle to materialize serving bundles "
                        "from the registry."
                    )
                },
            )
        final_status = (
            RealDbTrialStatus.ACCEPTED
            if commit_result.status is TaskRegistryCommitStatus.COMMITTED
            else RealDbTrialStatus.REGISTRY_DUPLICATE
        )
        summary = RealDbTrialSummary(
            db_id=db_id,
            requested_topic=topic,
            trial_status=final_status,
            flow_id=flow_id,
            phase_monitor_log_path=phase_monitor_log_path,
            debug_root=debug_root,
            debug_traces_dir=debug_traces_dir,
            synthesis_traces_dir=synthesis_traces_dir,
            solver_traces_dir=solver_traces_dir,
            task_id=draft.task_bundle.task_id,
            quality_gate_status="accept",
            solver_pass_rate=draft.task_bundle.quality_metrics.solver_pass_rate,
            solver_ci_low=draft.task_bundle.quality_metrics.solver_ci_low,
            solver_ci_high=draft.task_bundle.quality_metrics.solver_ci_high,
            solver_matched_runs=draft.task_bundle.quality_metrics.solver_matched_runs,
            solver_planned_runs=draft.task_bundle.quality_metrics.solver_planned_runs,
            solver_completed_runs=draft.task_bundle.quality_metrics.solver_completed_runs,
            solver_evaluable_runs=draft.task_bundle.quality_metrics.solver_evaluable_runs,
            solver_failed_runs=draft.task_bundle.quality_metrics.solver_failed_runs,
            registry_status=commit_result.status,
            registry_task_id=commit_result.task_id,
            bundle_root=bundle_root,
            elapsed_seconds=elapsed_seconds(),
        )
        phase_monitor.close()
        event_logger.close()
        return summary

    async def close(self) -> None:
        if self.synthesis_runtime is not None:
            await self.synthesis_runtime.close()
        if self._owns_solver_orchestrator and self.solver_orchestrator is not None:
            await self.solver_orchestrator.close()
            self.solver_orchestrator = None
            self._owns_solver_orchestrator = False
        if self._owns_registry:
            close_registry = getattr(self.registry, "close", None)
            if callable(close_registry):
                close_registry()

    def _synthesis_runtime_for_trial(
        self,
        debug_traces_dir: Path,
        phase_monitor: PipelinePhaseMonitorLogger,
        event_logger: TrialEventLogger | None = None,
    ) -> SynthesisAgentRuntime:
        if self.synthesis_runtime is not None:
            return self.synthesis_runtime
        runtime = SynthesisAgentRuntime(
            self.config.model_copy(
                update={
                    "output": self.config.output.model_copy(
                        update={"traces_dir": cast(Path, debug_traces_dir)}
                    )
                }
            ),
            phase_monitor=phase_monitor,
            database_pools=self.database_pools,
            solver_orchestrator=self.solver_orchestrator,
            synthesis_db=self.synthesis_db,
            synthesis_backends=self.synthesis_backends,
            event_logger=event_logger,
        )
        self.synthesis_runtime = runtime
        return runtime


def _encode_backend_failures(
    backend_failures: tuple[SynthesisBackendFailure, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{failure.provider}/{failure.model}:{failure.error_type}" for failure in backend_failures
    )
