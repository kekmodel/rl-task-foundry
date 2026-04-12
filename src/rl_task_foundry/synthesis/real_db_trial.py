"""Real-database single-environment trial runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.pipeline.environment_orchestrator import (
    EnvironmentOrchestrator,
    EnvironmentQualityGateStatus,
    evaluate_rollout_summary,
)
from rl_task_foundry.synthesis.bundle_exporter import EnvironmentBundleExporter
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.cross_instance import evaluate_cross_instance_draft
from rl_task_foundry.synthesis.environment_registry import (
    EnvironmentRegistryCommitStatus,
    EnvironmentRegistryWriter,
)
from rl_task_foundry.synthesis.pipeline_events import PipelineFlowLogger, build_flow_id
from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisBackendFailure,
    SynthesisPhaseExecutionError,
    SynthesisProviderUnavailableError,
    SynthesisRuntimeError,
    SynthesisSelfConsistencyError,
)


class RealDbTrialStatus(StrEnum):
    ACCEPTED = "accepted"
    REGISTRY_DUPLICATE = "registry_duplicate"
    REJECT_CROSS_INSTANCE = "reject_cross_instance"
    REJECT_TOO_HARD = "reject_too_hard"
    REJECT_TOO_EASY = "reject_too_easy"
    SYNTHESIS_FAILED = "synthesis_failed"


@dataclass(frozen=True, slots=True)
class RealDbTrialSummary:
    db_id: str
    requested_category: CategoryTaxonomy
    trial_status: RealDbTrialStatus
    summary_path: Path
    flow_id: str | None = None
    event_log_path: Path | None = None
    debug_root: Path | None = None
    debug_traces_dir: Path | None = None
    synthesis_traces_dir: Path | None = None
    solver_traces_dir: Path | None = None
    synthesis_session_db_path: Path | None = None
    solver_session_db_path: Path | None = None
    env_id: str | None = None
    quality_gate_status: str | None = None
    synthesis_error_type: str | None = None
    synthesis_error_message: str | None = None
    synthesis_phase: str | None = None
    backend_failures: tuple[str, ...] = ()
    attempt_outcomes: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    cross_instance_error_codes: tuple[str, ...] = ()
    solver_pass_rate: float | None = None
    solver_ci_low: float | None = None
    solver_ci_high: float | None = None
    registry_status: EnvironmentRegistryCommitStatus | None = None
    registry_env_id: str | None = None
    bundle_root: Path | None = None


@dataclass(slots=True)
class RealDbTrialRunner:
    config: AppConfig
    synthesis_runtime: SynthesisAgentRuntime | None = None
    environment_orchestrator: EnvironmentOrchestrator | None = None
    registry: EnvironmentRegistryWriter | None = None
    exporter: EnvironmentBundleExporter | None = None

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = EnvironmentRegistryWriter.for_config(self.config)
        if self.exporter is None:
            self.exporter = EnvironmentBundleExporter(
                registry=self.registry,
                materializer=self.registry.atomic_tool_materializer,
            )

    async def run(
        self,
        output_root: Path,
        *,
        db_id: str,
        category: CategoryTaxonomy,
    ) -> RealDbTrialSummary:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "trial_summary.json"
        debug_root = output_root / "debug"
        debug_traces_dir = debug_root / "traces"
        event_log_path = debug_root / "pipeline_events.jsonl"
        synthesis_traces_dir = debug_traces_dir / "synthesis"
        solver_traces_dir = debug_traces_dir
        synthesis_session_db_path = debug_traces_dir / "synthesis_sessions.sqlite"
        solver_session_db_path = debug_traces_dir / "sessions.sqlite"
        synthesis_traces_dir.mkdir(parents=True, exist_ok=True)
        flow_id = build_flow_id("real_db_trial")
        event_logger = PipelineFlowLogger(
            event_log_path=event_log_path,
            mirror_event_log_path=self.config.output.events_jsonl_path,
            flow_kind="real_db_trial",
            flow_id=flow_id,
        )
        event_logger.emit(
            stage="trial",
            status="started",
            payload={
                "db_id": db_id,
                "requested_category": category.value,
                "output_root": output_root,
                "summary_path": summary_path,
                "debug_root": debug_root,
            },
        )
        try:
            runtime = self._synthesis_runtime_for_trial(debug_traces_dir)
            event_logger.emit(
                stage="synthesis",
                status="started",
                payload={"db_id": db_id, "requested_category": category.value},
            )
            draft = await runtime.synthesize_environment_draft(
                db_id=db_id,
                requested_category=category,
            )
        except SynthesisSelfConsistencyError as exc:
            synthesis_failed_summary = self._synthesis_failure_summary(
                summary_path=summary_path,
                db_id=db_id,
                category=category,
                exc=exc,
                flow_id=flow_id,
                event_log_path=event_log_path,
                debug_root=debug_root,
                debug_traces_dir=debug_traces_dir,
                synthesis_traces_dir=synthesis_traces_dir,
                solver_traces_dir=solver_traces_dir,
                synthesis_session_db_path=synthesis_session_db_path,
                solver_session_db_path=solver_session_db_path,
            )
            event_logger.emit(
                stage="synthesis",
                status="failed",
                payload={
                    "error_type": synthesis_failed_summary.synthesis_error_type,
                    "error_message": synthesis_failed_summary.synthesis_error_message,
                    "attempt_outcomes": synthesis_failed_summary.attempt_outcomes,
                    "error_codes": synthesis_failed_summary.error_codes,
                },
            )
            event_logger.emit(
                stage="trial",
                status="completed",
                payload={"trial_status": synthesis_failed_summary.trial_status.value},
            )
            return self._write_summary(
                synthesis_failed_summary
            )
        except SynthesisPhaseExecutionError as exc:
            event_logger.emit(
                stage="synthesis",
                status="failed",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "phase": exc.phase.value if exc.phase is not None else None,
                    "backend_failures": _encode_backend_failures(exc.backend_failures),
                },
            )
            event_logger.emit(
                stage="trial",
                status="completed",
                payload={"trial_status": RealDbTrialStatus.SYNTHESIS_FAILED.value},
            )
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                    summary_path=summary_path,
                    flow_id=flow_id,
                    event_log_path=event_log_path,
                    debug_root=debug_root,
                    debug_traces_dir=debug_traces_dir,
                    synthesis_traces_dir=synthesis_traces_dir,
                    solver_traces_dir=solver_traces_dir,
                    synthesis_session_db_path=synthesis_session_db_path,
                    solver_session_db_path=solver_session_db_path,
                    synthesis_error_type=type(exc).__name__,
                    synthesis_error_message=str(exc),
                    synthesis_phase=exc.phase.value if exc.phase is not None else None,
                    backend_failures=_encode_backend_failures(exc.backend_failures),
                )
            )
        except SynthesisProviderUnavailableError as exc:
            event_logger.emit(
                stage="synthesis",
                status="failed",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "phase": exc.phase.value if exc.phase is not None else None,
                },
            )
            event_logger.emit(
                stage="trial",
                status="completed",
                payload={"trial_status": RealDbTrialStatus.SYNTHESIS_FAILED.value},
            )
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                    summary_path=summary_path,
                    flow_id=flow_id,
                    event_log_path=event_log_path,
                    debug_root=debug_root,
                    debug_traces_dir=debug_traces_dir,
                    synthesis_traces_dir=synthesis_traces_dir,
                    solver_traces_dir=solver_traces_dir,
                    synthesis_session_db_path=synthesis_session_db_path,
                    solver_session_db_path=solver_session_db_path,
                    synthesis_error_type=type(exc).__name__,
                    synthesis_error_message=str(exc),
                    synthesis_phase=exc.phase.value if exc.phase is not None else None,
                )
            )
        except SynthesisRuntimeError as exc:
            event_logger.emit(
                stage="synthesis",
                status="failed",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            event_logger.emit(
                stage="trial",
                status="completed",
                payload={"trial_status": RealDbTrialStatus.SYNTHESIS_FAILED.value},
            )
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                    summary_path=summary_path,
                    flow_id=flow_id,
                    event_log_path=event_log_path,
                    debug_root=debug_root,
                    debug_traces_dir=debug_traces_dir,
                    synthesis_traces_dir=synthesis_traces_dir,
                    solver_traces_dir=solver_traces_dir,
                    synthesis_session_db_path=synthesis_session_db_path,
                    solver_session_db_path=solver_session_db_path,
                    synthesis_error_type=type(exc).__name__,
                    synthesis_error_message=str(exc),
                )
            )

        event_logger.emit(
            stage="synthesis",
            status="completed",
            payload={
                "env_id": draft.environment.env_id,
                "selected_category": draft.selected_category.value,
                "payload_repair_codes": draft.self_consistency_diagnostics.payload_repair_codes,
            },
        )
        event_logger.emit(
            stage="cross_instance",
            status="started",
            payload={"env_id": draft.environment.env_id},
        )
        cross_instance_summary = evaluate_cross_instance_draft(draft)
        if not cross_instance_summary.passed:
            event_logger.emit(
                stage="cross_instance",
                status="failed",
                payload={
                    "env_id": draft.environment.env_id,
                    "error_codes": list(cross_instance_summary.error_codes),
                },
            )
            event_logger.emit(
                stage="trial",
                status="completed",
                payload={"trial_status": RealDbTrialStatus.REJECT_CROSS_INSTANCE.value},
            )
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=RealDbTrialStatus.REJECT_CROSS_INSTANCE,
                    summary_path=summary_path,
                    flow_id=flow_id,
                    event_log_path=event_log_path,
                    debug_root=debug_root,
                    debug_traces_dir=debug_traces_dir,
                    synthesis_traces_dir=synthesis_traces_dir,
                    solver_traces_dir=solver_traces_dir,
                    synthesis_session_db_path=synthesis_session_db_path,
                    solver_session_db_path=solver_session_db_path,
                    env_id=draft.environment.env_id,
                    cross_instance_error_codes=cross_instance_summary.error_codes,
                )
            )

        event_logger.emit(
            stage="cross_instance",
            status="passed",
            payload={"env_id": draft.environment.env_id},
        )
        environment_orchestrator = self._environment_orchestrator_for_trial(debug_traces_dir)
        event_logger.emit(
            stage="rollout",
            status="started",
            payload={"env_id": draft.environment.env_id},
        )
        rollout_summary = await environment_orchestrator.run_draft(draft)
        event_logger.emit(
            stage="rollout",
            status="completed",
            payload={
                "env_id": draft.environment.env_id,
                "planned_solver_runs": rollout_summary.planned_solver_runs,
                "total_solver_runs": rollout_summary.total_solver_runs,
                "matched_solver_runs": rollout_summary.matched_solver_runs,
                "early_stop_decision": rollout_summary.early_stop_decision,
            },
        )
        quality_gate_summary = evaluate_rollout_summary(self.config, rollout_summary)
        event_logger.emit(
            stage="quality_gate",
            status=quality_gate_summary.status.value,
            payload={
                "env_id": draft.environment.env_id,
                "pass_rate": quality_gate_summary.pass_rate,
                "ci_low": quality_gate_summary.ci_lower,
                "ci_high": quality_gate_summary.ci_upper,
            },
        )
        if quality_gate_summary.status is not EnvironmentQualityGateStatus.ACCEPT:
            event_logger.emit(
                stage="trial",
                status="completed",
                payload={
                    "trial_status": _quality_gate_trial_status(
                        quality_gate_summary.status
                    ).value
                },
            )
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=_quality_gate_trial_status(quality_gate_summary.status),
                    summary_path=summary_path,
                    flow_id=flow_id,
                    event_log_path=event_log_path,
                    debug_root=debug_root,
                    debug_traces_dir=debug_traces_dir,
                    synthesis_traces_dir=synthesis_traces_dir,
                    solver_traces_dir=solver_traces_dir,
                    synthesis_session_db_path=synthesis_session_db_path,
                    solver_session_db_path=solver_session_db_path,
                    env_id=draft.environment.env_id,
                    quality_gate_status=quality_gate_summary.status.value,
                    solver_pass_rate=quality_gate_summary.pass_rate,
                    solver_ci_low=quality_gate_summary.ci_lower,
                    solver_ci_high=quality_gate_summary.ci_upper,
                )
            )

        accepted_draft = accepted_draft_with_quality_metrics(
            draft,
            quality_gate_summary=quality_gate_summary,
        )
        event_logger.emit(
            stage="registry_commit",
            status="started",
            payload={"env_id": accepted_draft.environment.env_id},
        )
        commit_result = self.registry.commit_draft(accepted_draft)
        event_logger.emit(
            stage="registry_commit",
            status=commit_result.status.value,
            payload={
                "env_id": accepted_draft.environment.env_id,
                "registry_env_id": commit_result.env_id,
            },
        )
        bundle_root = output_root / "bundle"
        event_logger.emit(
            stage="bundle_export",
            status="started",
            payload={"bundle_root": bundle_root, "env_id": commit_result.env_id},
        )
        self.exporter.export_bundle(bundle_root, env_id=commit_result.env_id)
        event_logger.emit(
            stage="bundle_export",
            status="completed",
            payload={"bundle_root": bundle_root, "env_id": commit_result.env_id},
        )
        final_status = (
            RealDbTrialStatus.ACCEPTED
            if commit_result.status is EnvironmentRegistryCommitStatus.COMMITTED
            else RealDbTrialStatus.REGISTRY_DUPLICATE
        )
        event_logger.emit(
            stage="trial",
            status="completed",
            payload={"trial_status": final_status.value},
        )
        return self._write_summary(
            RealDbTrialSummary(
                db_id=db_id,
                requested_category=category,
                trial_status=final_status,
                summary_path=summary_path,
                flow_id=flow_id,
                event_log_path=event_log_path,
                debug_root=debug_root,
                debug_traces_dir=debug_traces_dir,
                synthesis_traces_dir=synthesis_traces_dir,
                solver_traces_dir=solver_traces_dir,
                synthesis_session_db_path=synthesis_session_db_path,
                solver_session_db_path=solver_session_db_path,
                env_id=accepted_draft.environment.env_id,
                quality_gate_status=quality_gate_summary.status.value,
                solver_pass_rate=quality_gate_summary.pass_rate,
                solver_ci_low=quality_gate_summary.ci_lower,
                solver_ci_high=quality_gate_summary.ci_upper,
                registry_status=commit_result.status,
                registry_env_id=commit_result.env_id,
                bundle_root=bundle_root,
            )
        )

    async def close(self) -> None:
        if self.synthesis_runtime is not None:
            await self.synthesis_runtime.close()
        if self.environment_orchestrator is not None:
            await self.environment_orchestrator.close()

    def _synthesis_failure_summary(
        self,
        *,
        summary_path: Path,
        db_id: str,
        category: CategoryTaxonomy,
        exc: SynthesisSelfConsistencyError,
        flow_id: str,
        event_log_path: Path,
        debug_root: Path,
        debug_traces_dir: Path,
        synthesis_traces_dir: Path,
        solver_traces_dir: Path,
        synthesis_session_db_path: Path,
        solver_session_db_path: Path,
    ) -> RealDbTrialSummary:
        error_codes: list[str] = []
        if exc.last_registration_diagnostics is not None:
            error_codes.extend(exc.last_registration_diagnostics.error_codes)
        if exc.last_self_consistency_diagnostics is not None:
            error_codes.extend(exc.last_self_consistency_diagnostics.error_codes)
        return RealDbTrialSummary(
            db_id=db_id,
            requested_category=category,
            trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
            summary_path=summary_path,
            flow_id=flow_id,
            event_log_path=event_log_path,
            debug_root=debug_root,
            debug_traces_dir=debug_traces_dir,
            synthesis_traces_dir=synthesis_traces_dir,
            solver_traces_dir=solver_traces_dir,
            synthesis_session_db_path=synthesis_session_db_path,
            solver_session_db_path=solver_session_db_path,
            synthesis_error_type=type(exc).__name__,
            synthesis_error_message=str(exc),
            attempt_outcomes=tuple(attempt.outcome.value for attempt in exc.attempts),
            error_codes=tuple(_dedupe_preserving_order(error_codes)),
        )

    @staticmethod
    def _summary_payload(summary: RealDbTrialSummary) -> dict[str, object]:
        payload = asdict(summary)
        payload["requested_category"] = summary.requested_category.value
        payload["trial_status"] = summary.trial_status.value
        payload["summary_path"] = str(summary.summary_path)
        for key in (
            "event_log_path",
            "debug_root",
            "debug_traces_dir",
            "synthesis_traces_dir",
            "solver_traces_dir",
            "synthesis_session_db_path",
            "solver_session_db_path",
        ):
            value = payload[key]
            payload[key] = str(value) if value is not None else None
        payload["registry_status"] = (
            summary.registry_status.value if summary.registry_status is not None else None
        )
        payload["bundle_root"] = str(summary.bundle_root) if summary.bundle_root is not None else None
        return payload

    def _write_summary(self, summary: RealDbTrialSummary) -> RealDbTrialSummary:
        summary.summary_path.write_text(
            json.dumps(
                self._summary_payload(summary),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return summary

    def _synthesis_runtime_for_trial(self, debug_traces_dir: Path) -> SynthesisAgentRuntime:
        if self.synthesis_runtime is None:
            trial_config = _config_with_trial_traces_dir(self.config, debug_traces_dir)
            self.synthesis_runtime = SynthesisAgentRuntime(trial_config)
        return cast(SynthesisAgentRuntime, self.synthesis_runtime)

    def _environment_orchestrator_for_trial(
        self,
        debug_traces_dir: Path,
    ) -> EnvironmentOrchestrator:
        if self.environment_orchestrator is None:
            trial_config = _config_with_trial_traces_dir(self.config, debug_traces_dir)
            self.environment_orchestrator = EnvironmentOrchestrator(trial_config)
        return cast(EnvironmentOrchestrator, self.environment_orchestrator)


def _quality_gate_trial_status(
    status: EnvironmentQualityGateStatus,
) -> RealDbTrialStatus:
    if status is EnvironmentQualityGateStatus.REJECT_TOO_HARD:
        return RealDbTrialStatus.REJECT_TOO_HARD
    if status is EnvironmentQualityGateStatus.REJECT_TOO_EASY:
        return RealDbTrialStatus.REJECT_TOO_EASY
    raise ValueError(f"unsupported non-accept quality gate status: {status}")


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _config_with_trial_traces_dir(config: AppConfig, traces_dir: Path) -> AppConfig:
    output = config.output.model_copy(
        update={
            "traces_dir": traces_dir,
        },
        deep=True,
    )
    return config.model_copy(update={"output": output}, deep=True)


def _encode_backend_failures(
    failures: tuple[SynthesisBackendFailure, ...] | list[SynthesisBackendFailure],
) -> tuple[str, ...]:
    return tuple(f"{failure.provider}/{failure.model}:{failure.error_type}" for failure in failures)
