"""Static registration policy checks for generated synthesis artifacts."""

from __future__ import annotations

import ast
from enum import StrEnum
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from rl_task_foundry.config.models import RegistrationPolicyConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactKind(StrEnum):
    TOOL_MODULE = "tool_module"
    SOLUTION_MODULE = "solution_module"
    VERIFIER_MODULE = "verifier_module"
    SHADOW_VERIFIER_MODULE = "shadow_verifier_module"


class RegistrationError(StrictModel):
    code: str
    line: int | None = None
    col: int | None = None
    node_type: str | None = None
    detail: str
    suggestion: str | None = None


def validate_generated_module(
    source: str,
    *,
    kind: ArtifactKind,
    policy: RegistrationPolicyConfig,
) -> list[RegistrationError]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            RegistrationError(
                code="syntax_error",
                line=exc.lineno,
                col=exc.offset,
                node_type="SyntaxError",
                detail=exc.msg,
                suggestion="Fix the generated Python syntax before registration.",
            )
        ]

    visitor = _PolicyVisitor(policy)
    visitor.visit(tree)
    errors = visitor.errors[:]
    errors.extend(_validate_module_signature(tree, kind=kind, policy=policy))
    return errors


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, policy: RegistrationPolicyConfig) -> None:
        self.policy = policy
        self.errors: list[RegistrationError] = []
        self._imported_roots: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            self._imported_roots.add(root)
            if root in self.policy.forbidden_module_roots:
                self._error(
                    node,
                    "forbidden_import",
                    f"Import of module '{root}' is not allowed.",
                    "Use only allowlisted standard-library imports in generated code.",
                )
            elif root not in self.policy.allowed_import_roots:
                self._error(
                    node,
                    "import_not_allowlisted",
                    f"Import root '{root}' is not in the registration allowlist.",
                    "Remove the import or extend the registration allowlist intentionally.",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level != 0:
            self._error(
                node,
                "relative_import_forbidden",
                "Relative imports are not allowed in generated code.",
                "Import only absolute allowlisted modules or runtime-provided facades.",
            )
            return
        module = node.module or ""
        root = module.split(".", 1)[0]
        self._imported_roots.add(root)
        if root in self.policy.forbidden_module_roots:
            self._error(
                node,
                "forbidden_import",
                f"Import of module '{root}' is not allowed.",
                "Use only allowlisted standard-library imports in generated code.",
            )
        elif root not in self.policy.allowed_import_roots:
            self._error(
                node,
                "import_not_allowlisted",
                f"Import root '{root}' is not in the registration allowlist.",
                "Remove the import or extend the registration allowlist intentionally.",
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._error(
            node,
            "class_definition_forbidden",
            "Dynamic class definitions are not allowed in generated code.",
            "Use plain functions and data literals only.",
        )

    def visit_With(self, node: ast.With) -> None:
        self._error(
            node,
            "with_statement_forbidden",
            "Context managers are not allowed in generated code.",
            "Use the runtime-provided tool facade instead of custom resource management.",
        )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._error(
            node,
            "async_with_forbidden",
            "Async context managers are not allowed in generated code.",
            "Use the runtime-provided tool facade instead of custom resource management.",
        )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self._error(
            node,
            "global_forbidden",
            "global statements are not allowed in generated code.",
            "Keep generated code side-effect free and function-local.",
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._error(
            node,
            "nonlocal_forbidden",
            "nonlocal statements are not allowed in generated code.",
            "Keep generated code side-effect free and function-local.",
        )

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._error(
            node,
            "lambda_forbidden",
            "Lambda expressions are not allowed in generated code.",
            "Use explicit named helper functions only if needed.",
        )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.policy.forbid_dunder_access and "__" in node.attr:
            self._error(
                node,
                "dunder_access_forbidden",
                f"Dunder attribute access '{node.attr}' is not allowed.",
                "Avoid reflection and Python object model escape hatches.",
            )
        if node.attr == "connect" and isinstance(node.value, ast.Name):
            if node.value.id in self.policy.forbidden_module_roots:
                self._error(
                    node,
                    "raw_db_connect_forbidden",
                    f"Direct database connection via '{node.value.id}.connect' is not allowed.",
                    "Use runtime-injected DB access only.",
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.policy.forbidden_symbols:
            self._error(
                node,
                "forbidden_symbol",
                f"Use of symbol '{node.id}' is not allowed.",
                "Remove reflective or unsafe builtins from generated code.",
            )
        if self.policy.forbid_dunder_access and "__" in node.id:
            self._error(
                node,
                "dunder_name_forbidden",
                f"Dunder-like name '{node.id}' is not allowed.",
                "Avoid reflection and Python object model escape hatches.",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if call_name in self.policy.forbidden_symbols:
            self._error(
                node,
                "forbidden_call",
                f"Call to '{call_name}' is not allowed.",
                "Use only allowlisted runtime helpers.",
            )
        if call_name.endswith(".connect"):
            root = call_name.split(".", 1)[0]
            if root in self.policy.forbidden_module_roots:
                self._error(
                    node,
                    "raw_db_connect_forbidden",
                    f"Direct database connection via '{call_name}' is not allowed.",
                    "Use runtime-injected DB access only.",
                )
        self.generic_visit(node)

    def _error(
        self,
        node: ast.AST,
        code: str,
        detail: str,
        suggestion: str | None = None,
    ) -> None:
        self.errors.append(
            RegistrationError(
                code=code,
                line=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                node_type=type(node).__name__,
                detail=detail,
                suggestion=suggestion,
            )
        )


def _validate_module_signature(
    tree: ast.Module,
    *,
    kind: ArtifactKind,
    policy: RegistrationPolicyConfig,
) -> list[RegistrationError]:
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    public_functions = [node for node in functions if not node.name.startswith("_")]
    errors: list[RegistrationError] = []

    if kind is ArtifactKind.TOOL_MODULE:
        if not public_functions:
            return [
                RegistrationError(
                    code="missing_tool_function",
                    detail="Tool modules must declare at least one public tool function.",
                    suggestion="Add an async tool function with signature async def tool_name(conn, ...).",
                )
            ]
        for function in public_functions:
            if policy.enforce_async_tool_functions and not isinstance(function, ast.AsyncFunctionDef):
                errors.append(
                    RegistrationError(
                        code="tool_must_be_async",
                        line=function.lineno,
                        col=function.col_offset,
                        node_type=type(function).__name__,
                        detail=f"Tool function '{function.name}' must be async.",
                        suggestion="Declare generated tools as async def tool_name(conn, ...).",
                    )
                )
            if not _arg_names(function) or _arg_names(function)[0] != "conn":
                errors.append(
                    RegistrationError(
                        code="tool_signature_invalid",
                        line=function.lineno,
                        col=function.col_offset,
                        node_type=type(function).__name__,
                        detail=f"Tool function '{function.name}' must accept 'conn' as its first argument.",
                        suggestion="Use signature async def tool_name(conn, **kwargs).",
                    )
                )
        return errors

    if kind is ArtifactKind.SOLUTION_MODULE:
        return _require_named_function(
            public_functions,
            expected_name="solve",
            expected_async=False,
            expected_args=["tools"],
            missing_code="missing_solve_function",
            invalid_code="solve_signature_invalid",
        )

    if kind in {ArtifactKind.VERIFIER_MODULE, ArtifactKind.SHADOW_VERIFIER_MODULE}:
        errors.extend(
            _require_named_function(
                public_functions,
                expected_name="verify",
                expected_async=False,
                expected_args=["answer", "tools"],
                missing_code="missing_verify_function",
                invalid_code="verify_signature_invalid",
            )
        )
        errors.extend(
            _require_named_function(
                public_functions,
                expected_name="fetch_facts",
                expected_async=False,
                expected_args=["answer", "tools"],
                missing_code="missing_fetch_facts_function",
                invalid_code="fetch_facts_signature_invalid",
            )
        )
        errors.extend(
            _require_named_function(
                public_functions,
                expected_name="facts_match_answer_claims",
                expected_async=False,
                expected_args=["answer", "facts"],
                missing_code="missing_facts_match_function",
                invalid_code="facts_match_signature_invalid",
            )
        )
        errors.extend(
            _require_named_function(
                public_functions,
                expected_name="check_constraints",
                expected_async=False,
                expected_args=["answer", "facts"],
                missing_code="missing_check_constraints_function",
                invalid_code="check_constraints_signature_invalid",
            )
        )
        return errors

    return errors


def _require_named_function(
    functions: Iterable[ast.FunctionDef | ast.AsyncFunctionDef],
    *,
    expected_name: str,
    expected_async: bool,
    expected_args: list[str],
    missing_code: str,
    invalid_code: str,
) -> list[RegistrationError]:
    matches = [function for function in functions if function.name == expected_name]
    if not matches:
        return [
            RegistrationError(
                code=missing_code,
                detail=f"Missing required function '{expected_name}'.",
                suggestion=f"Define {expected_name}({', '.join(expected_args)}).",
            )
        ]
    function = matches[0]
    errors: list[RegistrationError] = []
    if expected_async != isinstance(function, ast.AsyncFunctionDef):
        errors.append(
            RegistrationError(
                code=invalid_code,
                line=function.lineno,
                col=function.col_offset,
                node_type=type(function).__name__,
                detail=f"Function '{expected_name}' has the wrong async/sync form.",
                suggestion=(
                    f"Declare {'async ' if expected_async else ''}def {expected_name}("
                    f"{', '.join(expected_args)})."
                ),
            )
        )
    if _arg_names(function) != expected_args:
        errors.append(
            RegistrationError(
                code=invalid_code,
                line=function.lineno,
                col=function.col_offset,
                node_type=type(function).__name__,
                detail=(
                    f"Function '{expected_name}' must use arguments {expected_args}, "
                    f"got {_arg_names(function)}."
                ),
                suggestion=f"Use signature def {expected_name}({', '.join(expected_args)}).",
            )
        )
    return errors


def _arg_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [arg.arg for arg in function.args.args]


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        current: ast.AST | None = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return ".".join(parts)
    return "<unknown>"
