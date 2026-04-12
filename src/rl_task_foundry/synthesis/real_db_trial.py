"""Real-database single-environment trial runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

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
from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
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
    env_id: str | None = None
    quality_gate_status: str | None = None
    synthesis_error_type: str | None = None
    synthesis_error_message: str | None = None
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
    synthesis_runtime: Any | None = None
    environment_orchestrator: Any | None = None
    registry: EnvironmentRegistryWriter | None = None
    exporter: EnvironmentBundleExporter | None = None

    def __post_init__(self) -> None:
        if self.synthesis_runtime is None:
            self.synthesis_runtime = SynthesisAgentRuntime(self.config)
        if self.environment_orchestrator is None:
            self.environment_orchestrator = EnvironmentOrchestrator(self.config)
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
        try:
            draft = await self.synthesis_runtime.synthesize_environment_draft(
                db_id=db_id,
                requested_category=category,
            )
        except SynthesisSelfConsistencyError as exc:
            return self._write_summary(
                self._synthesis_failure_summary(
                    summary_path=summary_path,
                    db_id=db_id,
                    category=category,
                    exc=exc,
                )
            )
        except SynthesisRuntimeError as exc:
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=RealDbTrialStatus.SYNTHESIS_FAILED,
                    summary_path=summary_path,
                    synthesis_error_type=type(exc).__name__,
                    synthesis_error_message=str(exc),
                )
            )

        cross_instance_summary = evaluate_cross_instance_draft(draft)
        if not cross_instance_summary.passed:
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=RealDbTrialStatus.REJECT_CROSS_INSTANCE,
                    summary_path=summary_path,
                    env_id=draft.environment.env_id,
                    cross_instance_error_codes=cross_instance_summary.error_codes,
                )
            )

        rollout_summary = await self.environment_orchestrator.run_draft(draft)
        quality_gate_summary = evaluate_rollout_summary(self.config, rollout_summary)
        if quality_gate_summary.status is not EnvironmentQualityGateStatus.ACCEPT:
            return self._write_summary(
                RealDbTrialSummary(
                    db_id=db_id,
                    requested_category=category,
                    trial_status=_quality_gate_trial_status(quality_gate_summary.status),
                    summary_path=summary_path,
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
        commit_result = self.registry.commit_draft(accepted_draft)
        bundle_root = output_root / "bundle"
        self.exporter.export_bundle(bundle_root, env_id=commit_result.env_id)
        return self._write_summary(
            RealDbTrialSummary(
                db_id=db_id,
                requested_category=category,
                trial_status=(
                    RealDbTrialStatus.ACCEPTED
                    if commit_result.status is EnvironmentRegistryCommitStatus.COMMITTED
                    else RealDbTrialStatus.REGISTRY_DUPLICATE
                ),
                summary_path=summary_path,
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
