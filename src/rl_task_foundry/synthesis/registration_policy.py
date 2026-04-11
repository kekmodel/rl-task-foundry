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
    TOOL_SELF_TEST_MODULE = "tool_self_test_module"
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
    errors = _validate_top_level_statements(tree)
    errors.extend(visitor.errors)
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
                "Use only absolute allowlisted modules and runtime-provided function arguments.",
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
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._error(
            node,
            "with_statement_forbidden",
            "Context managers are not allowed in generated code.",
            "Use runtime-provided function arguments like 'tools' instead of custom resource management.",
        )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._error(
            node,
            "async_with_forbidden",
            "Async context managers are not allowed in generated code.",
            "Use runtime-provided function arguments like 'tools' instead of custom resource management.",
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
        if _is_annotation_node(node):
            return
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

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = _literal_subscript_key(node.slice)
        if isinstance(key, str):
            if self.policy.forbid_dunder_access and "__" in key:
                self._error(
                    node,
                    "dunder_subscript_forbidden",
                    f"Dunder-like subscript key '{key}' is not allowed.",
                    "Avoid reflective or Python object model escape-hatch keys.",
                )
            if _is_call_target(node) and key in self.policy.forbidden_symbols:
                self._error(
                    node,
                    "forbidden_subscript_call",
                    f"Calling a value loaded from subscript key '{key}' is not allowed.",
                    "Use runtime-provided function arguments instead of reflective lookup.",
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

    def generic_visit(self, node: ast.AST) -> None:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        setattr(item, "_policy_parent", node)
                        setattr(item, "_policy_parent_field", field)
                        self.visit(item)
            elif isinstance(value, ast.AST):
                setattr(value, "_policy_parent", node)
                setattr(value, "_policy_parent_field", field)
                self.visit(value)

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
    errors.extend(_duplicate_function_errors(public_functions))

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
        errors.extend(
            _require_named_function(
                public_functions,
                expected_name="solve",
                expected_async=False,
                expected_args=["tools"],
                missing_code="missing_solve_function",
                invalid_code="solve_signature_invalid",
            )
        )
        return errors

    if kind is ArtifactKind.TOOL_SELF_TEST_MODULE:
        errors.extend(
            _require_named_function(
                public_functions,
                expected_name="run_self_test",
                expected_async=True,
                expected_args=["tools"],
                missing_code="missing_run_self_test_function",
                invalid_code="run_self_test_signature_invalid",
            )
        )
        return errors

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
    if len(matches) > 1:
        return [
            RegistrationError(
                code=f"duplicate_{expected_name}_function",
                line=function.lineno,
                col=function.col_offset,
                node_type=type(function).__name__,
                detail=f"Function '{expected_name}' must be defined exactly once.",
                suggestion=f"Keep a single definition of {expected_name}({', '.join(expected_args)}).",
            )
            for function in matches[1:]
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
    if isinstance(node, ast.Subscript):
        base = _call_name(node.value)
        key = _literal_subscript_key(node.slice)
        if key is None:
            return f"{base}[<subscript>]"
        return f"{base}[{key!r}]"
    return "<unknown>"


def _validate_top_level_statements(tree: ast.Module) -> list[RegistrationError]:
    errors: list[RegistrationError] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        errors.append(
            RegistrationError(
                code="top_level_statement_forbidden",
                line=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                node_type=type(node).__name__,
                detail="Top-level executable statements are not allowed in generated modules.",
                suggestion="Move executable logic inside the required entrypoint functions.",
            )
        )
    return errors


def _duplicate_function_errors(
    functions: Iterable[ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[RegistrationError]:
    seen: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    errors: list[RegistrationError] = []
    for function in functions:
        previous = seen.get(function.name)
        if previous is not None:
            errors.append(
                RegistrationError(
                    code="duplicate_public_function",
                    line=function.lineno,
                    col=function.col_offset,
                    node_type=type(function).__name__,
                    detail=f"Public function '{function.name}' is defined more than once.",
                    suggestion="Keep a single public definition per function name.",
                )
            )
            continue
        seen[function.name] = function
    return errors


def _literal_subscript_key(node: ast.AST) -> str | int | float | bool | None:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (str, int, float, bool)):
            return value
    return None


def _is_annotation_node(node: ast.AST) -> bool:
    current: ast.AST | None = node
    while current is not None:
        parent = getattr(current, "_policy_parent", None)
        field = getattr(current, "_policy_parent_field", None)
        if field in {"annotation", "returns"}:
            return True
        current = parent
    return False


def _is_call_target(node: ast.AST) -> bool:
    parent = getattr(node, "_policy_parent", None)
    field = getattr(node, "_policy_parent_field", None)
    return isinstance(parent, ast.Call) and field == "func"
