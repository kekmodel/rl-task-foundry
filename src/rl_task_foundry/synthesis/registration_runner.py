"""High-level registration runner for generated synthesis artifacts."""

from __future__ import annotations

import ast
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.contracts import MaterializedFactsSchema
from rl_task_foundry.synthesis.registration_policy import (
    ArtifactKind,
    RegistrationError,
    VerifierHybridAnalysis,
    analyze_verifier_module,
    validate_generated_module,
)
from rl_task_foundry.synthesis.subprocess_pool import (
    RegistrationExecutionResult,
    RegistrationSubprocessPool,
    RegistrationVerifierProbeResult,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegistrationArtifactName(StrEnum):
    TOOL = "tool"
    TOOL_SELF_TEST = "tool_self_test"
    SOLUTION = "solution"
    VERIFIER = "verifier"
    SHADOW_VERIFIER = "shadow_verifier"


class RegistrationBundleStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class GeneratedArtifactBundle(StrictModel):
    solution_source: str
    verifier_source: str
    shadow_verifier_source: str


class VerifierProbeSpec(StrictModel):
    answer_sample: object
    facts_schema: MaterializedFactsSchema


class ArtifactRegistrationResult(StrictModel):
    artifact_name: RegistrationArtifactName
    artifact_kind: ArtifactKind
    static_errors: list[RegistrationError] = Field(default_factory=list)
    verifier_hybrid_analysis: VerifierHybridAnalysis | None = None
    execution_required: bool = False
    executed: bool = False
    execution_errors: list[RegistrationError] = Field(default_factory=list)
    execution_call_count: int | None = None
    execution_return_value: object | None = None
    probe_required: bool = False
    probe_executed: bool = False
    probe_errors: list[RegistrationError] = Field(default_factory=list)
    verifier_execution_probe: RegistrationVerifierProbeResult | None = None

    @property
    def passed(self) -> bool:
        return self.static_passed and self.runtime_passed and self.probe_passed

    @property
    def static_passed(self) -> bool:
        return not self.static_errors

    @property
    def runtime_passed(self) -> bool:
        if not self.execution_required:
            return True
        return self.executed and not self.execution_errors

    @property
    def probe_passed(self) -> bool:
        if not self.probe_required:
            return True
        return self.probe_executed and not self.probe_errors


class RegistrationArtifactDiagnostics(StrictModel):
    artifact_name: RegistrationArtifactName
    passed: bool
    static_passed: bool
    runtime_passed: bool
    probe_passed: bool
    execution_required: bool = False
    executed: bool = False
    probe_required: bool = False
    probe_executed: bool = False
    static_error_codes: list[str] = Field(default_factory=list)
    execution_error_codes: list[str] = Field(default_factory=list)
    probe_error_codes: list[str] = Field(default_factory=list)
    weak_signal_codes: list[str] = Field(default_factory=list)
    verify_stage_calls: dict[str, int] = Field(default_factory=dict)
    fetch_facts_tool_calls: int | None = None
    fetch_facts_answer_references: int | None = None
    facts_match_answer_references: int | None = None
    facts_match_facts_references: int | None = None
    check_constraints_facts_references: int | None = None
    facts_match_constant_boolean_return: bool | None = None
    check_constraints_constant_boolean_return: bool | None = None
    probe_fetch_facts_return_keys: list[str] = Field(default_factory=list)
    probe_expected_fact_keys: list[str] = Field(default_factory=list)
    probe_missing_fact_keys: list[str] = Field(default_factory=list)
    probe_extra_fact_keys: list[str] = Field(default_factory=list)
    probe_fetch_facts_tool_calls: int | None = None
    probe_fetch_facts_answer_reads: int | None = None
    probe_facts_match_answer_reads: int | None = None
    probe_facts_match_facts_reads: int | None = None
    probe_check_constraints_facts_reads: int | None = None
    probe_verify_tool_calls: int | None = None
    probe_facts_match_result: bool | None = None
    probe_check_constraints_result: bool | None = None
    probe_verify_result: bool | None = None


class RegistrationBundleDiagnostics(StrictModel):
    status: RegistrationBundleStatus
    failing_artifacts: list[RegistrationArtifactName] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    weak_signal_codes: list[str] = Field(default_factory=list)
    tool: RegistrationArtifactDiagnostics
    tool_self_test: RegistrationArtifactDiagnostics
    solution: RegistrationArtifactDiagnostics
    verifier: RegistrationArtifactDiagnostics
    shadow_verifier: RegistrationArtifactDiagnostics


class RegistrationBundleReport(StrictModel):
    status: RegistrationBundleStatus
    tool: ArtifactRegistrationResult
    tool_self_test: ArtifactRegistrationResult
    solution: ArtifactRegistrationResult
    verifier: ArtifactRegistrationResult
    shadow_verifier: ArtifactRegistrationResult


def build_registration_diagnostics(
    report: RegistrationBundleReport,
) -> RegistrationBundleDiagnostics:
    tool = _artifact_diagnostics(report.tool)
    tool_self_test = _artifact_diagnostics(report.tool_self_test)
    solution = _artifact_diagnostics(report.solution)
    verifier = _artifact_diagnostics(report.verifier)
    shadow_verifier = _artifact_diagnostics(report.shadow_verifier)
    artifacts = [tool, tool_self_test, solution, verifier, shadow_verifier]
    return RegistrationBundleDiagnostics(
        status=report.status,
        failing_artifacts=[artifact.artifact_name for artifact in artifacts if not artifact.passed],
        error_codes=_dedupe_preserving_order(
            [
                *tool.static_error_codes,
                *tool.execution_error_codes,
                *tool.probe_error_codes,
                *tool_self_test.static_error_codes,
                *tool_self_test.execution_error_codes,
                *tool_self_test.probe_error_codes,
                *solution.static_error_codes,
                *solution.execution_error_codes,
                *solution.probe_error_codes,
                *verifier.static_error_codes,
                *verifier.execution_error_codes,
                *verifier.probe_error_codes,
                *shadow_verifier.static_error_codes,
                *shadow_verifier.execution_error_codes,
                *shadow_verifier.probe_error_codes,
            ]
        ),
        weak_signal_codes=_dedupe_preserving_order(
            [
                *tool.weak_signal_codes,
                *tool_self_test.weak_signal_codes,
                *solution.weak_signal_codes,
                *verifier.weak_signal_codes,
                *shadow_verifier.weak_signal_codes,
            ]
        ),
        tool=tool,
        tool_self_test=tool_self_test,
        solution=solution,
        verifier=verifier,
        shadow_verifier=shadow_verifier,
    )


def _artifact_diagnostics(result: ArtifactRegistrationResult) -> RegistrationArtifactDiagnostics:
    analysis = result.verifier_hybrid_analysis
    probe = result.verifier_execution_probe
    return RegistrationArtifactDiagnostics(
        artifact_name=result.artifact_name,
        passed=result.passed,
        static_passed=result.static_passed,
        runtime_passed=result.runtime_passed,
        probe_passed=result.probe_passed,
        execution_required=result.execution_required,
        executed=result.executed,
        probe_required=result.probe_required,
        probe_executed=result.probe_executed,
        static_error_codes=[error.code for error in result.static_errors],
        execution_error_codes=[error.code for error in result.execution_errors],
        probe_error_codes=[error.code for error in result.probe_errors],
        weak_signal_codes=_weak_signal_codes(result),
        verify_stage_calls=dict(analysis.verify_stage_calls) if analysis is not None else {},
        fetch_facts_tool_calls=analysis.fetch_facts_tool_calls if analysis is not None else None,
        fetch_facts_answer_references=analysis.fetch_facts_answer_references
        if analysis is not None
        else None,
        facts_match_answer_references=analysis.facts_match_answer_references
        if analysis is not None
        else None,
        facts_match_facts_references=analysis.facts_match_facts_references
        if analysis is not None
        else None,
        check_constraints_facts_references=analysis.check_constraints_facts_references
        if analysis is not None
        else None,
        facts_match_constant_boolean_return=analysis.facts_match_constant_boolean_return
        if analysis is not None
        else None,
        check_constraints_constant_boolean_return=analysis.check_constraints_constant_boolean_return
        if analysis is not None
        else None,
        probe_fetch_facts_return_keys=list(probe.fetch_facts_return_keys)
        if probe is not None
        else [],
        probe_expected_fact_keys=list(probe.expected_fact_keys) if probe is not None else [],
        probe_missing_fact_keys=list(probe.missing_fact_keys) if probe is not None else [],
        probe_extra_fact_keys=list(probe.extra_fact_keys) if probe is not None else [],
        probe_fetch_facts_tool_calls=probe.fetch_facts_tool_calls if probe is not None else None,
        probe_fetch_facts_answer_reads=probe.fetch_facts_answer_reads
        if probe is not None
        else None,
        probe_facts_match_answer_reads=probe.facts_match_answer_reads
        if probe is not None
        else None,
        probe_facts_match_facts_reads=probe.facts_match_facts_reads
        if probe is not None
        else None,
        probe_check_constraints_facts_reads=probe.check_constraints_facts_reads
        if probe is not None
        else None,
        probe_verify_tool_calls=probe.verify_tool_calls if probe is not None else None,
        probe_facts_match_result=probe.facts_match_result if probe is not None else None,
        probe_check_constraints_result=probe.check_constraints_result if probe is not None else None,
        probe_verify_result=probe.verify_result if probe is not None else None,
    )


def _weak_signal_codes(result: ArtifactRegistrationResult) -> list[str]:
    analysis = result.verifier_hybrid_analysis
    probe = result.verifier_execution_probe
    weak_signals: list[str] = []
    if analysis is not None:
        if analysis.fetch_facts_tool_calls == 0:
            weak_signals.append("fetch_facts_missing_tool_usage")
        if analysis.fetch_facts_answer_references == 0:
            weak_signals.append("fetch_facts_missing_answer_usage")
        if analysis.facts_match_answer_references == 0:
            weak_signals.append("facts_match_missing_answer_usage")
        if analysis.facts_match_facts_references == 0:
            weak_signals.append("facts_match_missing_facts_usage")
        if analysis.check_constraints_facts_references == 0:
            weak_signals.append("check_constraints_missing_facts_usage")
        if analysis.facts_match_constant_boolean_return:
            weak_signals.append("facts_match_constant_boolean")
        if analysis.check_constraints_constant_boolean_return:
            weak_signals.append("check_constraints_constant_boolean")
    if probe is not None:
        if probe.fetch_facts_tool_calls == 0:
            weak_signals.append("probe_fetch_facts_missing_tool_usage")
        if probe.fetch_facts_answer_reads == 0:
            weak_signals.append("probe_fetch_facts_missing_answer_usage")
        if probe.facts_match_answer_reads == 0:
            weak_signals.append("probe_facts_match_missing_answer_usage")
        if probe.facts_match_facts_reads == 0:
            weak_signals.append("probe_facts_match_missing_facts_usage")
        if probe.check_constraints_facts_reads == 0:
            weak_signals.append("probe_check_constraints_missing_facts_usage")
        if probe.missing_fact_keys or probe.extra_fact_keys:
            weak_signals.append("probe_facts_schema_key_drift")
    return _dedupe_preserving_order(weak_signals)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _sources_are_structurally_identical(left: str, right: str) -> bool:
    try:
        left_tree = ast.parse(left)
        right_tree = ast.parse(right)
    except SyntaxError:
        return left.strip() == right.strip()
    return ast.dump(left_tree, include_attributes=False) == ast.dump(
        right_tree,
        include_attributes=False,
    )


def _shadow_independence_errors(bundle: GeneratedArtifactBundle) -> list[RegistrationError]:
    if not _sources_are_structurally_identical(
        bundle.verifier_source,
        bundle.shadow_verifier_source,
    ):
        return []
    return [
        RegistrationError(
            code="shadow_verifier_not_independent",
            detail=(
                "shadow_verifier_source must not be structurally identical to "
                "verifier_source."
            ),
            suggestion=(
                "Use a meaningfully different shadow verification path instead of "
                "reusing the primary verifier."
            ),
        )
    ]


async def run_registration_bundle(
    *,
    config: AppConfig,
    bundle: GeneratedArtifactBundle,
    tool_source: str | None = None,
    atomic_tool_set_ref: str | None = None,
    database_execution_config: dict[str, object] | None = None,
    tool_self_test_source: str | None = None,
    pool: RegistrationSubprocessPool | None = None,
    verifier_probe_specs: dict[RegistrationArtifactName, VerifierProbeSpec] | None = None,
    answer_field_names: set[str] | None = None,
) -> RegistrationBundleReport:
    """Run Milestone 2 registration checks for a generated artifact bundle.

    Passing an existing pool is the normal production path. Letting the runner create
    its own pool is primarily a convenience for tests and one-off local checks.
    """

    policy = config.synthesis.registration_policy
    tool_definitions = _load_atomic_tool_definitions(
        config=config,
        atomic_tool_set_ref=atomic_tool_set_ref,
    )
    tool = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.TOOL,
        artifact_kind=ArtifactKind.TOOL_MODULE,
        static_errors=(
            validate_generated_module(
                tool_source,
                kind=ArtifactKind.TOOL_MODULE,
                policy=policy,
            )
            if tool_source is not None
            else []
        ),
    )
    tool_self_test = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.TOOL_SELF_TEST,
        artifact_kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
        execution_required=(
            tool_self_test_source is not None
            and (tool_source is not None or atomic_tool_set_ref is not None)
        ),
        static_errors=(
            validate_generated_module(
                tool_self_test_source,
                kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
                policy=policy,
            )
            if tool_self_test_source is not None
            else []
        ),
    )
    solution = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.SOLUTION,
        artifact_kind=ArtifactKind.SOLUTION_MODULE,
        static_errors=[
            *validate_generated_module(
                bundle.solution_source,
                kind=ArtifactKind.SOLUTION_MODULE,
                policy=policy,
            ),
            *_validate_tool_contract_usage(
                bundle.solution_source,
                tool_definitions=tool_definitions,
            ),
        ],
    )
    verifier = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.VERIFIER,
        artifact_kind=ArtifactKind.VERIFIER_MODULE,
        probe_required=verifier_probe_specs is not None
        and RegistrationArtifactName.VERIFIER in verifier_probe_specs,
        static_errors=[
            *validate_generated_module(
                bundle.verifier_source,
                kind=ArtifactKind.VERIFIER_MODULE,
                policy=policy,
            ),
            *_validate_tool_contract_usage(
                bundle.verifier_source,
                tool_definitions=tool_definitions,
            ),
            *_validate_answer_contract_usage(
                bundle.verifier_source,
                answer_field_names=answer_field_names,
            ),
        ],
        verifier_hybrid_analysis=analyze_verifier_module(
            bundle.verifier_source,
            kind=ArtifactKind.VERIFIER_MODULE,
        ),
    )
    shadow_verifier = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.SHADOW_VERIFIER,
        artifact_kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
        probe_required=verifier_probe_specs is not None
        and RegistrationArtifactName.SHADOW_VERIFIER in verifier_probe_specs,
        static_errors=[
            *validate_generated_module(
                bundle.shadow_verifier_source,
                kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
                policy=policy,
            ),
            *_validate_tool_contract_usage(
                bundle.shadow_verifier_source,
                tool_definitions=tool_definitions,
            ),
            *_validate_answer_contract_usage(
                bundle.shadow_verifier_source,
                answer_field_names=answer_field_names,
            ),
            *_shadow_independence_errors(bundle),
        ],
        verifier_hybrid_analysis=analyze_verifier_module(
            bundle.shadow_verifier_source,
            kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
        ),
    )

    owns_pool = pool is None
    if tool.static_passed and tool_self_test.static_passed and tool_self_test.execution_required:
        pool = pool or await RegistrationSubprocessPool.start(config)
        execution_result = await pool.run_tool_self_test(
            tool_source=tool_source,
            atomic_tool_set_ref=atomic_tool_set_ref,
            database_execution_config=database_execution_config,
            self_test_source=tool_self_test_source,
        )
        tool_self_test.executed = True
        tool_self_test.execution_errors = execution_result.errors
        tool_self_test.execution_call_count = execution_result.call_count
        tool_self_test.execution_return_value = execution_result.return_value

    if verifier_probe_specs is not None and tool.static_passed:
        if tool_source is None and atomic_tool_set_ref is None:
            raise ValueError(
                "tool_source or atomic_tool_set_ref is required when verifier probe specs are provided"
            )
        pool = pool or await RegistrationSubprocessPool.start(config)
        for artifact_result, artifact_name, verifier_source, artifact_kind in (
            (
                verifier,
                RegistrationArtifactName.VERIFIER,
                bundle.verifier_source,
                ArtifactKind.VERIFIER_MODULE,
            ),
            (
                shadow_verifier,
                RegistrationArtifactName.SHADOW_VERIFIER,
                bundle.shadow_verifier_source,
                ArtifactKind.SHADOW_VERIFIER_MODULE,
            ),
        ):
            probe_spec = verifier_probe_specs.get(artifact_name)
            if probe_spec is None or not artifact_result.static_passed:
                continue
            probe_result = await pool.probe_verifier_module(
                tool_source=tool_source,
                atomic_tool_set_ref=atomic_tool_set_ref,
                database_execution_config=database_execution_config,
                verifier_source=verifier_source,
                artifact_kind=artifact_kind,
                answer_sample=probe_spec.answer_sample,
                expected_fact_keys=[fact.key for fact in probe_spec.facts_schema.facts],
            )
            artifact_result.probe_executed = True
            artifact_result.probe_errors = probe_result.errors
            artifact_result.verifier_execution_probe = probe_result

    if owns_pool and pool is not None:
        await pool.close()

    status = (
        RegistrationBundleStatus.PASSED
        if all(
            result.passed
            for result in [tool, tool_self_test, solution, verifier, shadow_verifier]
        )
        else RegistrationBundleStatus.FAILED
    )
    return RegistrationBundleReport(
        status=status,
        tool=tool,
        tool_self_test=tool_self_test,
        solution=solution,
        verifier=verifier,
        shadow_verifier=shadow_verifier,
    )


def _load_atomic_tool_definitions(
    *,
    config: AppConfig,
    atomic_tool_set_ref: str | None,
) -> dict[str, dict[str, object]]:
    if not isinstance(atomic_tool_set_ref, str) or not atomic_tool_set_ref.startswith("db://"):
        return {}
    db_id = atomic_tool_set_ref.removeprefix("db://")
    payload = AtomicToolMaterializer.for_config(config).read_actor_tool_definitions(db_id=db_id)
    tool_definitions: dict[str, dict[str, object]] = {}
    for item in payload:
        name = item.get("name")
        if isinstance(name, str):
            tool_definitions[name] = item
    return tool_definitions


def _validate_tool_contract_usage(
    source: str,
    *,
    tool_definitions: dict[str, dict[str, object]],
) -> list[RegistrationError]:
    if not tool_definitions:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _ToolContractVisitor(tool_definitions)
    visitor.visit(tree)
    return visitor.errors


def _validate_answer_contract_usage(
    source: str,
    *,
    answer_field_names: set[str] | None,
) -> list[RegistrationError]:
    if not answer_field_names:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _AnswerContractVisitor(answer_field_names)
    visitor.visit(tree)
    return visitor.errors


class _ToolContractVisitor(ast.NodeVisitor):
    def __init__(self, tool_definitions: dict[str, dict[str, object]]) -> None:
        self.tool_definitions = tool_definitions
        self.errors: list[RegistrationError] = []
        self._bindings: dict[str, tuple[str, dict[str, object]]] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            binding = self._binding_for_expr(node.value)
            if binding is None:
                self._bindings.pop(target_name, None)
            else:
                self._bindings[target_name] = binding
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            target_name = node.target.id
            binding = self._binding_for_expr(node.value) if node.value is not None else None
            if binding is None:
                self._bindings.pop(target_name, None)
            else:
                self._bindings[target_name] = binding
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            binding = self._binding_for_expr(node.func.value)
            if binding is not None:
                tool_name, tool_definition = binding
                self._validate_key_access(
                    node,
                    key=node.args[0].value,
                    tool_name=tool_name,
                    tool_definition=tool_definition,
                    via_method="get",
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            binding = self._binding_for_expr(node.value)
            if binding is not None:
                tool_name, tool_definition = binding
                self._validate_key_access(
                    node,
                    key=node.slice.value,
                    tool_name=tool_name,
                    tool_definition=tool_definition,
                    via_method="subscript",
                )
        self.generic_visit(node)

    def _binding_for_expr(self, expr: ast.AST | None) -> tuple[str, dict[str, object]] | None:
        if expr is None:
            return None
        if isinstance(expr, ast.Name):
            return self._bindings.get(expr.id)
        if isinstance(expr, ast.Call):
            tool_name = _tools_attribute_name(expr.func)
            if tool_name is None:
                return None
            tool_definition = self.tool_definitions.get(tool_name)
            if tool_definition is None:
                return None
            return tool_name, tool_definition
        if (
            isinstance(expr, ast.Subscript)
            and isinstance(expr.slice, ast.Constant)
            and isinstance(expr.slice.value, int)
        ):
            binding = self._binding_for_expr(expr.value)
            if binding is None:
                return None
            tool_name, tool_definition = binding
            if _returns_row_objects(tool_definition.get("returns_schema")):
                return tool_name, tool_definition
        return None

    def _validate_key_access(
        self,
        node: ast.AST,
        *,
        key: str,
        tool_name: str,
        tool_definition: dict[str, object],
        via_method: str,
    ) -> None:
        allowed_keys = _allowed_return_keys(tool_definition.get("returns_schema"))
        if allowed_keys is None:
            self.errors.append(
                RegistrationError(
                    code="tool_return_key_access_forbidden",
                    line=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    node_type=type(node).__name__,
                    detail=(
                        f"{via_method} access to key '{key}' is invalid because tool "
                        f"'{tool_name}' does not return an object schema with named fields."
                    ),
                    suggestion=(
                        "Only index into object-valued tool results whose returns_schema "
                        "declares that field explicitly."
                    ),
                )
            )
            return
        if key in allowed_keys:
            return
        sorted_keys = ", ".join(sorted(allowed_keys)) or "<none>"
        self.errors.append(
            RegistrationError(
                code="tool_return_key_not_in_returns_schema",
                line=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                node_type=type(node).__name__,
                detail=(
                    f"Tool '{tool_name}' does not declare return key '{key}'. "
                    f"Allowed keys: {sorted_keys}."
                ),
                suggestion=(
                    "Re-read available_atomic_tools and derive hidden attributes through "
                    "other atomic tools instead of assuming undeclared return keys exist."
                ),
            )
        )


class _AnswerContractVisitor(ast.NodeVisitor):
    def __init__(self, answer_field_names: set[str]) -> None:
        self.answer_field_names = answer_field_names
        self.errors: list[RegistrationError] = []

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "answer"
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self._validate_answer_key(node, node.args[0].value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "answer"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            self._validate_answer_key(node, node.slice.value)
        self.generic_visit(node)

    def _validate_answer_key(self, node: ast.AST, key: str) -> None:
        if key in self.answer_field_names:
            return
        allowed_keys = ", ".join(sorted(self.answer_field_names)) or "<none>"
        self.errors.append(
            RegistrationError(
                code="answer_key_not_in_output_schema",
                line=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                node_type=type(node).__name__,
                detail=(
                    f"Generated verifier code reads answer key '{key}', but output_schema "
                    f"declares only: {allowed_keys}."
                ),
                suggestion=(
                    "Read only declared answer keys. Recompute hidden anchors or comparator "
                    "entities through atomic tools instead of expecting them in the submitted answer."
                ),
            )
        )


def _tools_attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tools":
        return node.attr
    return None


def _allowed_return_keys(returns_schema: object) -> set[str] | None:
    if not isinstance(returns_schema, dict):
        return None
    schema_type = returns_schema.get("type")
    if schema_type == "object":
        properties = returns_schema.get("properties")
        if isinstance(properties, dict):
            return {key for key in properties if isinstance(key, str)}
    if schema_type == "array":
        items = returns_schema.get("items")
        if isinstance(items, dict):
            return _allowed_return_keys(items)
    any_of = returns_schema.get("anyOf")
    if isinstance(any_of, list):
        for option in any_of:
            allowed = _allowed_return_keys(option)
            if allowed is not None:
                return allowed
    return None


def _returns_row_objects(returns_schema: object) -> bool:
    if not isinstance(returns_schema, dict):
        return False
    return returns_schema.get("type") == "array" and _allowed_return_keys(returns_schema) is not None
