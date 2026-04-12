"""Real-database single-environment trial runner."""

from __future__ import annotations

import json
import inspect
from dataclasses import asdict, dataclass, replace
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
    build_semantic_dedup_text,
    difficulty_vector_json,
    estimate_semantic_similarity,
)
from rl_task_foundry.synthesis.phase_monitor import (
    PipelinePhaseMonitorLogger,
)
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisBackendFailure,
    SynthesisDifficultyRetrySeed,
    SynthesisEnvironmentDraft,
    SynthesisPhaseExecutionError,
    SynthesisProviderUnavailableError,
    SynthesisQualityGateFeedback,
    SynthesisRuntimeError,
    SynthesisSelfConsistencyError,
    merge_strongest_difficulty_vector,
)

QUALITY_RETRY_SEMANTIC_SIMILARITY_THRESHOLD = 0.98


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
    phase_monitor_log_path: Path | None = None
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
    quality_retry_count: int = 0
    quality_retry_history: tuple[str, ...] = ()
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
        phase_monitor_log_path = debug_root / "phase_monitors.jsonl"
        synthesis_traces_dir = debug_traces_dir / "synthesis"
        solver_traces_dir = debug_traces_dir
        synthesis_session_db_path = debug_traces_dir / "synthesis_sessions.sqlite"
        solver_session_db_path = debug_traces_dir / "sessions.sqlite"
        synthesis_traces_dir.mkdir(parents=True, exist_ok=True)
        flow_id = build_flow_id("real_db_trial")
        phase_monitor = PipelinePhaseMonitorLogger(
            phase_monitor_log_path=phase_monitor_log_path,
            flow_kind="real_db_trial",
            flow_id=flow_id,
        )
        runtime = self._synthesis_runtime_for_trial(debug_traces_dir, phase_monitor)
        retry_seed = SynthesisDifficultyRetrySeed()

        while True:
            if (
                retry_seed.retry_requires_harder
                and retry_seed.difficulty_crank_index
                >= self.config.synthesis.runtime.max_difficulty_cranks
            ):
                phase_monitor.emit(
                    phase="quality_retry",
                    status="limit_exhausted",
                    expected_contract={
                        "max_difficulty_cranks": self.config.synthesis.runtime.max_difficulty_cranks,
                    },
                    actual_data={
                        "quality_retry_count": retry_seed.difficulty_crank_index,
                        "quality_retry_history": [
                            axis.value for axis in retry_seed.difficulty_crank_history
                        ],
                    },
                    checks={"can_retry": False},
                    diagnostics={
                        "latest_quality_gate_feedback": (
                            retry_seed.latest_quality_gate_feedback.model_dump(mode="json")
                            if retry_seed.latest_quality_gate_feedback is not None
                            else None
                        )
                    },
                )
                return self._write_summary(
                    RealDbTrialSummary(
                        db_id=db_id,
                        requested_category=category,
                        trial_status=RealDbTrialStatus.REJECT_TOO_EASY,
                        summary_path=summary_path,
                        flow_id=flow_id,
                        phase_monitor_log_path=phase_monitor_log_path,
                        debug_root=debug_root,
                        debug_traces_dir=debug_traces_dir,
                        synthesis_traces_dir=synthesis_traces_dir,
                        solver_traces_dir=solver_traces_dir,
                        synthesis_session_db_path=synthesis_session_db_path,
                        solver_session_db_path=solver_session_db_path,
                        env_id=(
                            retry_seed.latest_quality_gate_feedback.previous_env_id
                            if retry_seed.latest_quality_gate_feedback is not None
                            else None
                        ),
                        quality_gate_status="reject_too_easy",
                        solver_pass_rate=(
                            retry_seed.latest_quality_gate_feedback.pass_rate
                            if retry_seed.latest_quality_gate_feedback is not None
                            else None
                        ),
                        solver_ci_low=(
                            retry_seed.latest_quality_gate_feedback.ci_lower
                            if retry_seed.latest_quality_gate_feedback is not None
                            else None
                        ),
                        solver_ci_high=(
                            retry_seed.latest_quality_gate_feedback.ci_upper
                            if retry_seed.latest_quality_gate_feedback is not None
                            else None
                        ),
                        quality_retry_count=retry_seed.difficulty_crank_index,
                        quality_retry_history=tuple(
                            axis.value for axis in retry_seed.difficulty_crank_history
                        ),
                        error_codes=("quality_retry_budget_exhausted",),
                    )
                )

            active_retry_seed = retry_seed if retry_seed.retry_requires_harder else None
            if active_retry_seed is not None:
                phase_monitor.emit(
                    phase="quality_retry",
                    status="request_harder",
                    expected_contract={
                        "max_difficulty_cranks": self.config.synthesis.runtime.max_difficulty_cranks,
                    },
                    actual_data={
                        "quality_retry_count": active_retry_seed.difficulty_crank_index,
                        "quality_retry_history": [
                            axis.value for axis in active_retry_seed.difficulty_crank_history
                        ],
                        "next_crank_axis": active_retry_seed.requested_axis().value,
                    },
                    checks={"retry_requires_harder": True},
                    diagnostics={
                        "latest_quality_gate_feedback": (
                            active_retry_seed.latest_quality_gate_feedback.model_dump(mode="json")
                            if active_retry_seed.latest_quality_gate_feedback is not None
                            else None
                        )
                    },
                )
            try:
                draft = await runtime.synthesize_environment_draft(
                    db_id=db_id,
                    requested_category=category,
                    retry_seed=active_retry_seed,
                )
            except SynthesisSelfConsistencyError as exc:
                consumed_retry_seed = (
                    active_retry_seed.consume_requested_crank()
                    if active_retry_seed is not None
                    else retry_seed
                )
                synthesis_failed_summary = self._synthesis_failure_summary(
                    summary_path=summary_path,
                    db_id=db_id,
                    category=category,
                    exc=exc,
                    flow_id=flow_id,
                    phase_monitor_log_path=phase_monitor_log_path,
                    debug_root=debug_root,
                    debug_traces_dir=debug_traces_dir,
                    synthesis_traces_dir=synthesis_traces_dir,
                    solver_traces_dir=solver_traces_dir,
                    synthesis_session_db_path=synthesis_session_db_path,
                    solver_session_db_path=solver_session_db_path,
                )
                return self._write_summary(
                    replace(
                        synthesis_failed_summary,
                        quality_retry_count=consumed_retry_seed.difficulty_crank_index,
                        quality_retry_history=tuple(
                            axis.value for axis in consumed_retry_seed.difficulty_crank_history
                        ),
                    )
                )
            except SynthesisPhaseExecutionError as exc:
                consumed_retry_seed = (
                    active_retry_seed.consume_requested_crank()
                    if active_retry_seed is not None
                    else retry_seed
                )
                return self._write_summary(
                    RealDbTrialSummary(
                        db_id=db_id,
                        requested_category=category,
                        trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                        summary_path=summary_path,
                        flow_id=flow_id,
                        phase_monitor_log_path=phase_monitor_log_path,
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
                        quality_retry_count=consumed_retry_seed.difficulty_crank_index,
                        quality_retry_history=tuple(
                            axis.value for axis in consumed_retry_seed.difficulty_crank_history
                        ),
                    )
                )
            except SynthesisProviderUnavailableError as exc:
                consumed_retry_seed = (
                    active_retry_seed.consume_requested_crank()
                    if active_retry_seed is not None
                    else retry_seed
                )
                return self._write_summary(
                    RealDbTrialSummary(
                        db_id=db_id,
                        requested_category=category,
                        trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                        summary_path=summary_path,
                        flow_id=flow_id,
                        phase_monitor_log_path=phase_monitor_log_path,
                        debug_root=debug_root,
                        debug_traces_dir=debug_traces_dir,
                        synthesis_traces_dir=synthesis_traces_dir,
                        solver_traces_dir=solver_traces_dir,
                        synthesis_session_db_path=synthesis_session_db_path,
                        solver_session_db_path=solver_session_db_path,
                        synthesis_error_type=type(exc).__name__,
                        synthesis_error_message=str(exc),
                        synthesis_phase=exc.phase.value if exc.phase is not None else None,
                        quality_retry_count=consumed_retry_seed.difficulty_crank_index,
                        quality_retry_history=tuple(
                            axis.value for axis in consumed_retry_seed.difficulty_crank_history
                        ),
                    )
                )
            except SynthesisRuntimeError as exc:
                consumed_retry_seed = (
                    active_retry_seed.consume_requested_crank()
                    if active_retry_seed is not None
                    else retry_seed
                )
                return self._write_summary(
                    RealDbTrialSummary(
                        db_id=db_id,
                        requested_category=category,
                        trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                        summary_path=summary_path,
                        flow_id=flow_id,
                        phase_monitor_log_path=phase_monitor_log_path,
                        debug_root=debug_root,
                        debug_traces_dir=debug_traces_dir,
                        synthesis_traces_dir=synthesis_traces_dir,
                        solver_traces_dir=solver_traces_dir,
                        synthesis_session_db_path=synthesis_session_db_path,
                        solver_session_db_path=solver_session_db_path,
                        synthesis_error_type=type(exc).__name__,
                        synthesis_error_message=str(exc),
                        quality_retry_count=consumed_retry_seed.difficulty_crank_index,
                        quality_retry_history=tuple(
                            axis.value for axis in consumed_retry_seed.difficulty_crank_history
                        ),
                    )
                )

            if active_retry_seed is not None:
                retry_seed = active_retry_seed.consume_requested_crank()
            semantic_retry_diagnostics = _evaluate_quality_retry_semantic_change(
                draft=draft,
                feedback=retry_seed.latest_quality_gate_feedback,
            )
            if semantic_retry_diagnostics is not None and not semantic_retry_diagnostics["passed"]:
                phase_monitor.emit(
                    phase="quality_retry",
                    status="semantic_change_rejected",
                    expected_contract={
                        "must_change_latent_task": True,
                        "semantic_similarity_threshold": QUALITY_RETRY_SEMANTIC_SIMILARITY_THRESHOLD,
                    },
                    actual_data={
                        "env_id": draft.environment.env_id,
                        "question": draft.environment.task.question,
                        "rendered_user_prompt": (
                            draft.instances[0].rendered_user_prompt if draft.instances else None
                        ),
                        "difficulty_vector": difficulty_vector_json(
                            draft.environment.task.difficulty_vector
                        ),
                        "semantic_dedup_text": build_semantic_dedup_text(draft.environment),
                        "canonical_answer_jsons": [
                            answer.canonical_answer_json for answer in draft.canonical_answers
                        ],
                        "solution_fingerprints": [
                            answer.solution_fingerprint for answer in draft.canonical_answers
                        ],
                    },
                    checks=semantic_retry_diagnostics,
                    diagnostics={
                        "previous_env_id": (
                            retry_seed.latest_quality_gate_feedback.previous_env_id
                            if retry_seed.latest_quality_gate_feedback is not None
                            else None
                        ),
                    },
                )
                retry_seed = retry_seed.model_copy(
                    update={
                        "strongest_difficulty_vector": merge_strongest_difficulty_vector(
                            retry_seed.strongest_difficulty_vector,
                            draft.environment.task.difficulty_vector,
                        ),
                        "retry_requires_harder": True,
                        "latest_quality_gate_feedback": _quality_gate_feedback_from_draft(
                            draft=draft,
                            status="reject_too_easy",
                            pass_rate=(
                                retry_seed.latest_quality_gate_feedback.pass_rate
                                if retry_seed.latest_quality_gate_feedback is not None
                                else 1.0
                            ),
                            ci_lower=(
                                retry_seed.latest_quality_gate_feedback.ci_lower
                                if retry_seed.latest_quality_gate_feedback is not None
                                else 1.0
                            ),
                            ci_upper=(
                                retry_seed.latest_quality_gate_feedback.ci_upper
                                if retry_seed.latest_quality_gate_feedback is not None
                                else 1.0
                            ),
                            matched_solver_runs=(
                                retry_seed.latest_quality_gate_feedback.matched_solver_runs
                                if retry_seed.latest_quality_gate_feedback is not None
                                else 0
                            ),
                            total_solver_runs=(
                                retry_seed.latest_quality_gate_feedback.total_solver_runs
                                if retry_seed.latest_quality_gate_feedback is not None
                                else 0
                            ),
                        ),
                    }
                )
                continue
            cross_instance_summary = evaluate_cross_instance_draft(draft)
            phase_monitor.emit(
                phase="cross_instance",
                status="passed" if cross_instance_summary.passed else "failed",
                expected_contract={
                    "minimum_required": draft.environment.cross_instance_set.minimum_required,
                    "require_distinct_solution_fingerprints": (
                        draft.environment.cross_instance_set.require_distinct_solution_fingerprints
                    ),
                },
                actual_data={
                    "instance_count": cross_instance_summary.instance_count,
                    "canonical_answer_count": cross_instance_summary.canonical_answer_count,
                    "error_codes": list(cross_instance_summary.error_codes),
                },
                checks={"passed": cross_instance_summary.passed},
                diagnostics={"env_id": draft.environment.env_id},
            )
            if not cross_instance_summary.passed:
                return self._write_summary(
                    RealDbTrialSummary(
                        db_id=db_id,
                        requested_category=category,
                        trial_status=RealDbTrialStatus.REJECT_CROSS_INSTANCE,
                        summary_path=summary_path,
                        flow_id=flow_id,
                        phase_monitor_log_path=phase_monitor_log_path,
                        debug_root=debug_root,
                        debug_traces_dir=debug_traces_dir,
                        synthesis_traces_dir=synthesis_traces_dir,
                        solver_traces_dir=solver_traces_dir,
                        synthesis_session_db_path=synthesis_session_db_path,
                        solver_session_db_path=solver_session_db_path,
                        env_id=draft.environment.env_id,
                        cross_instance_error_codes=cross_instance_summary.error_codes,
                        quality_retry_count=retry_seed.difficulty_crank_index,
                        quality_retry_history=tuple(
                            axis.value for axis in retry_seed.difficulty_crank_history
                        ),
                    )
                )
            environment_orchestrator = self._environment_orchestrator_for_trial(debug_traces_dir)
            rollout_summary = await environment_orchestrator.run_draft(draft)
            phase_monitor.emit(
                phase="rollout",
                status="completed",
                expected_contract={
                    "planned_solver_runs_upper_bound": self.config.calibration.full_replica_limit,
                    "max_turns": draft.environment.rollout_constraints.max_turns,
                },
                actual_data={
                    "planned_solver_runs": rollout_summary.planned_solver_runs,
                    "total_solver_runs": rollout_summary.total_solver_runs,
                    "matched_solver_runs": rollout_summary.matched_solver_runs,
                    "early_stop_decision": rollout_summary.early_stop_decision,
                },
                checks={
                    "executed_runs_within_plan": rollout_summary.total_solver_runs
                    <= rollout_summary.planned_solver_runs,
                },
                diagnostics={"env_id": draft.environment.env_id},
            )
            quality_gate_summary = evaluate_rollout_summary(self.config, rollout_summary)
            phase_monitor.emit(
                phase="quality_gate",
                status=quality_gate_summary.status.value,
                expected_contract={
                    "band_lower": quality_gate_summary.band_lower,
                    "band_upper": quality_gate_summary.band_upper,
                },
                actual_data={
                    "pass_rate": quality_gate_summary.pass_rate,
                    "ci_low": quality_gate_summary.ci_lower,
                    "ci_high": quality_gate_summary.ci_upper,
                    "matched_solver_runs": quality_gate_summary.matched_solver_runs,
                    "total_solver_runs": quality_gate_summary.total_solver_runs,
                },
                checks={
                    "accepted": quality_gate_summary.status
                    is EnvironmentQualityGateStatus.ACCEPT,
                },
                diagnostics={"env_id": draft.environment.env_id},
            )
            if quality_gate_summary.status is not EnvironmentQualityGateStatus.ACCEPT:
                if quality_gate_summary.status is EnvironmentQualityGateStatus.REJECT_TOO_EASY:
                    retry_seed = retry_seed.model_copy(
                        update={
                            "strongest_difficulty_vector": merge_strongest_difficulty_vector(
                                retry_seed.strongest_difficulty_vector,
                                draft.environment.task.difficulty_vector,
                            ),
                            "retry_requires_harder": True,
                            "latest_quality_gate_feedback": _quality_gate_feedback_from_draft(
                                draft=draft,
                                status=quality_gate_summary.status.value,
                                pass_rate=quality_gate_summary.pass_rate,
                                ci_lower=quality_gate_summary.ci_lower,
                                ci_upper=quality_gate_summary.ci_upper,
                                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                                total_solver_runs=quality_gate_summary.total_solver_runs,
                            ),
                        }
                    )
                    continue
                return self._write_summary(
                    RealDbTrialSummary(
                        db_id=db_id,
                        requested_category=category,
                        trial_status=_quality_gate_trial_status(quality_gate_summary.status),
                        summary_path=summary_path,
                        flow_id=flow_id,
                        phase_monitor_log_path=phase_monitor_log_path,
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
                        quality_retry_count=retry_seed.difficulty_crank_index,
                        quality_retry_history=tuple(
                            axis.value for axis in retry_seed.difficulty_crank_history
                        ),
                    )
                )

            accepted_draft = accepted_draft_with_quality_metrics(
                draft,
                quality_gate_summary=quality_gate_summary,
            )
            commit_result = self.registry.commit_draft(accepted_draft)
            phase_monitor.emit(
                phase="registry_commit",
                status=commit_result.status.value,
                expected_contract={
                    "dedup_enabled": True,
                },
                actual_data={
                    "registry_env_id": commit_result.env_id,
                    "status": commit_result.status.value,
                },
                checks={
                    "committed_or_duplicate": commit_result.status
                    in {
                        EnvironmentRegistryCommitStatus.COMMITTED,
                        EnvironmentRegistryCommitStatus.DUPLICATE,
                    },
                },
                diagnostics={"env_id": accepted_draft.environment.env_id},
            )
            bundle_root = output_root / "bundle"
            self.exporter.export_bundle(bundle_root, env_id=commit_result.env_id)
            phase_monitor.emit(
                phase="bundle_export",
                status="completed",
                expected_contract={
                    "bundle_root": bundle_root,
                },
                actual_data={
                    "bundle_root": bundle_root,
                    "env_id": commit_result.env_id,
                },
                checks={"bundle_root_exists": bundle_root.exists()},
                diagnostics={},
            )
            final_status = (
                RealDbTrialStatus.ACCEPTED
                if commit_result.status is EnvironmentRegistryCommitStatus.COMMITTED
                else RealDbTrialStatus.REGISTRY_DUPLICATE
            )
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=final_status,
                    summary_path=summary_path,
                    flow_id=flow_id,
                    phase_monitor_log_path=phase_monitor_log_path,
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
                    quality_retry_count=retry_seed.difficulty_crank_index,
                    quality_retry_history=tuple(
                        axis.value for axis in retry_seed.difficulty_crank_history
                    ),
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
        phase_monitor_log_path: Path,
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
            phase_monitor_log_path=phase_monitor_log_path,
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
            "phase_monitor_log_path",
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

    def _synthesis_runtime_for_trial(
        self,
        debug_traces_dir: Path,
        phase_monitor: PipelinePhaseMonitorLogger,
    ) -> SynthesisAgentRuntime:
        if self.synthesis_runtime is None:
            trial_config = _config_with_trial_traces_dir(self.config, debug_traces_dir)
            runtime_signature = inspect.signature(SynthesisAgentRuntime)
            if "phase_monitor" in runtime_signature.parameters:
                self.synthesis_runtime = SynthesisAgentRuntime(
                    trial_config,
                    phase_monitor=phase_monitor,
                )
            else:
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


def _normalize_retry_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _quality_gate_feedback_from_draft(
    *,
    draft: SynthesisEnvironmentDraft,
    status: str,
    pass_rate: float,
    ci_lower: float,
    ci_upper: float,
    matched_solver_runs: int,
    total_solver_runs: int,
) -> SynthesisQualityGateFeedback:
    return SynthesisQualityGateFeedback(
        status=status,
        pass_rate=pass_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        matched_solver_runs=matched_solver_runs,
        total_solver_runs=total_solver_runs,
        previous_env_id=draft.environment.env_id,
        previous_question=draft.environment.task.question,
        previous_rendered_user_prompt=(
            draft.instances[0].rendered_user_prompt if draft.instances else None
        ),
        previous_semantic_dedup_text=build_semantic_dedup_text(draft.environment),
        previous_difficulty_vector=draft.environment.task.difficulty_vector,
        previous_canonical_answers=[
            answer.canonical_answer_json for answer in draft.canonical_answers
        ],
        previous_solution_fingerprints=[
            answer.solution_fingerprint for answer in draft.canonical_answers
        ],
    )


def _evaluate_quality_retry_semantic_change(
    *,
    draft: SynthesisEnvironmentDraft,
    feedback: SynthesisQualityGateFeedback | None,
) -> dict[str, object] | None:
    if feedback is None or feedback.status != EnvironmentQualityGateStatus.REJECT_TOO_EASY.value:
        return None

    current_question = _normalize_retry_text(draft.environment.task.question)
    current_rendered_prompt = _normalize_retry_text(
        draft.instances[0].rendered_user_prompt if draft.instances else None
    )
    previous_question = _normalize_retry_text(feedback.previous_question)
    previous_rendered_prompt = _normalize_retry_text(feedback.previous_rendered_user_prompt)
    current_semantic_dedup_text = build_semantic_dedup_text(draft.environment)
    previous_semantic_dedup_text = feedback.previous_semantic_dedup_text
    semantic_similarity = (
        estimate_semantic_similarity(
            previous_semantic_dedup_text,
            current_semantic_dedup_text,
        )
        if previous_semantic_dedup_text
        else None
    )
    same_question = (
        previous_question is not None and current_question == previous_question
    )
    same_rendered_user_prompt = (
        previous_rendered_prompt is not None
        and current_rendered_prompt == previous_rendered_prompt
    )
    same_semantic_dedup_text = (
        semantic_similarity is not None
        and semantic_similarity >= QUALITY_RETRY_SEMANTIC_SIMILARITY_THRESHOLD
    )
    current_canonical_answers = [
        answer.canonical_answer_json for answer in draft.canonical_answers
    ]
    current_solution_fingerprints = [
        answer.solution_fingerprint for answer in draft.canonical_answers
    ]
    same_canonical_answers = (
        bool(feedback.previous_canonical_answers)
        and current_canonical_answers == feedback.previous_canonical_answers
    )
    same_solution_fingerprints = (
        bool(feedback.previous_solution_fingerprints)
        and current_solution_fingerprints == feedback.previous_solution_fingerprints
    )
    passed = not (
        (same_question or same_rendered_user_prompt or same_semantic_dedup_text)
        and (same_canonical_answers or same_solution_fingerprints)
    )
    return {
        "passed": passed,
        "same_question": same_question,
        "same_rendered_user_prompt": same_rendered_user_prompt,
        "same_semantic_dedup_text": same_semantic_dedup_text,
        "semantic_similarity": semantic_similarity,
        "same_canonical_answers": same_canonical_answers,
        "same_solution_fingerprints": same_solution_fingerprints,
    }


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
