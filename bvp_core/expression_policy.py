"""Restricted, non-evaluating mathematical expression language for the BVP core."""

from __future__ import annotations

import ast
import keyword
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import sympy as sp

from .exceptions import ExpressionValidationError


class ExpressionContext(str, Enum):
    ODE = "ODE"
    BOUNDARY = "BOUNDARY"
    AUXILIARY = "AUXILIARY"
    SCALAR = "SCALAR"


@dataclass(frozen=True)
class ExpressionPolicy:
    max_characters: int = 512
    max_ast_nodes: int = 256
    max_nesting_depth: int = 32
    max_numeric_literal_digits: int = 32
    max_function_calls: int = 32
    max_power_operations: int = 16
    max_abs_exponent: float = 64.0


DEFAULT_EXPRESSION_POLICY = ExpressionPolicy()

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
)
ALLOWED_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)
ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "atan2": sp.atan2,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "sign": sp.sign,
}
ALLOWED_CONSTANTS = {"pi": sp.pi, "E": sp.E}
_FUNCTION_ARITIES = {
    **{name: (1,) for name in ALLOWED_FUNCTIONS if name not in {"atan2", "log"}},
    "atan2": (2,),
    "log": (1, 2),
}
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_BOUNDARY_ALIAS = re.compile(r"^x\d+_[0T]$")
_DECIMAL_LITERAL = re.compile(
    r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_RESERVED_IDENTIFIERS = (
    set(ALLOWED_FUNCTIONS) | set(ALLOWED_CONSTANTS) | {"t"}
)


def _context_value(context: ExpressionContext | str) -> str:
    return context.value if isinstance(context, ExpressionContext) else str(context)


def _raise_expression_error(
    *,
    context: ExpressionContext | str,
    expression: object,
    error_code: str,
    message: str,
    field_index: int | None = None,
    position: int | None = None,
) -> None:
    raise ExpressionValidationError(
        context=_context_value(context),
        expression=expression,
        error_code=error_code,
        message=message,
        field_index=field_index,
        position=position,
    )


def validate_identifiers(
    names: list[str] | tuple[str, ...],
    *,
    context: ExpressionContext | str,
) -> None:
    """Reject ambiguous, reserved, or non-ASCII state/output identifiers."""
    seen: set[str] = set()
    for index, name in enumerate(names):
        if (
            not isinstance(name, str)
            or not _IDENTIFIER.fullmatch(name)
            or keyword.iskeyword(name)
            or name.startswith("_")
            or "__" in name
        ):
            _raise_expression_error(
                context=context,
                expression=name,
                error_code="invalid_identifier",
                message=f"{name!r} is not an allowed identifier",
                field_index=index,
            )
        if name in seen:
            _raise_expression_error(
                context=context,
                expression=name,
                error_code="name_conflict",
                message=f"duplicate identifier {name!r}",
                field_index=index,
            )
        if name in _RESERVED_IDENTIFIERS or _BOUNDARY_ALIAS.fullmatch(name):
            _raise_expression_error(
                context=context,
                expression=name,
                error_code="name_conflict",
                message=f"identifier {name!r} is reserved by the expression language",
                field_index=index,
            )
        seen.add(name)


class _AstSafetyValidator:
    """Validate the complete AST before any SymPy object construction occurs."""

    def __init__(
        self,
        expression: str,
        *,
        context: ExpressionContext | str,
        allowed_symbols: Mapping[str, sp.Symbol],
        field_index: int | None,
        policy: ExpressionPolicy,
    ) -> None:
        self.expression = expression
        self.context = context
        self.allowed_symbols = allowed_symbols
        self.field_index = field_index
        self.policy = policy

    def error(self, node: ast.AST | None, code: str, message: str) -> None:
        position = getattr(node, "col_offset", None)
        if position is not None:
            position += 1
        _raise_expression_error(
            context=self.context,
            expression=self.expression,
            error_code=code,
            message=message,
            field_index=self.field_index,
            position=position,
        )

    def validate(self, tree: ast.Expression) -> None:
        nodes = list(ast.walk(tree))
        if len(nodes) > self.policy.max_ast_nodes:
            self.error(
                tree,
                "ast_too_complex",
                f"AST node count exceeds {self.policy.max_ast_nodes}",
            )
        calls = sum(isinstance(node, ast.Call) for node in nodes)
        if calls > self.policy.max_function_calls:
            self.error(
                tree,
                "function_call_limit",
                f"function call count exceeds {self.policy.max_function_calls}",
            )
        powers = sum(
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
            for node in nodes
        )
        if powers > self.policy.max_power_operations:
            self.error(
                tree,
                "power_operation_limit",
                f"power operation count exceeds {self.policy.max_power_operations}",
            )
        self._visit(tree.body, depth=1)

    def _visit(self, node: ast.AST, *, depth: int) -> None:
        if depth > self.policy.max_nesting_depth:
            self.error(
                node,
                "nesting_too_deep",
                f"expression nesting exceeds {self.policy.max_nesting_depth}",
            )
        if isinstance(node, ast.Constant):
            self._validate_constant(node)
            return
        if isinstance(node, ast.Name):
            self._validate_symbol_name(node)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ALLOWED_UNARY_OPERATORS):
                self.error(node, "disallowed_operator", "unary operator is not allowed")
            self._visit(node.operand, depth=depth + 1)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, ALLOWED_BINARY_OPERATORS):
                self.error(node, "disallowed_operator", "binary operator is not allowed")
            if isinstance(node.op, ast.Pow):
                exponent = self._constant_number(node.right)
                if exponent is None:
                    self.error(
                        node.right,
                        "power_exponent_not_constant",
                        "power exponent must be a numeric constant",
                    )
                if abs(exponent) > self.policy.max_abs_exponent:
                    self.error(
                        node.right,
                        "power_exponent_out_of_range",
                        "absolute power exponent exceeds "
                        f"{self.policy.max_abs_exponent:g}",
                    )
            self._visit(node.left, depth=depth + 1)
            self._visit(node.right, depth=depth + 1)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                self.error(
                    node.func,
                    "disallowed_call_target",
                    "only direct calls to allowed mathematical functions are permitted",
                )
            function_name = node.func.id
            if function_name.startswith("_") or "__" in function_name:
                self.error(node.func, "forbidden_name", "private names are forbidden")
            if function_name not in ALLOWED_FUNCTIONS:
                self.error(
                    node.func,
                    "unknown_function",
                    f"function {function_name!r} is not allowed; allowed functions: "
                    + ", ".join(sorted(ALLOWED_FUNCTIONS)),
                )
            if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
                self.error(
                    node,
                    "disallowed_call_arguments",
                    "keyword and starred arguments are not allowed",
                )
            if len(node.args) not in _FUNCTION_ARITIES[function_name]:
                allowed = ", ".join(str(value) for value in _FUNCTION_ARITIES[function_name])
                self.error(
                    node,
                    "disallowed_call_arguments",
                    f"{function_name} expects {allowed} positional argument(s)",
                )
            for argument in node.args:
                self._visit(argument, depth=depth + 1)
            return
        if isinstance(node, ast.Attribute):
            self.error(node, "attribute_access", "attribute access is forbidden")
        if isinstance(node, ast.Subscript):
            self.error(node, "subscript_access", "subscript access is forbidden")
        self.error(
            node,
            "disallowed_node",
            f"syntax node {type(node).__name__} is not allowed",
        )

    def _validate_constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.error(node, "invalid_literal", "only real numeric literals are allowed")
        if not math.isfinite(float(value)):
            self.error(node, "invalid_literal", "numeric literals must be finite")
        source = ast.get_source_segment(self.expression, node) or repr(value)
        if not _DECIMAL_LITERAL.fullmatch(source):
            self.error(
                node,
                "invalid_literal",
                "only decimal integer, decimal fraction, and scientific notation "
                "literals are allowed",
            )
        digit_count = sum(character.isdigit() for character in source)
        if digit_count > self.policy.max_numeric_literal_digits:
            self.error(
                node,
                "numeric_literal_too_long",
                "numeric literal exceeds "
                f"{self.policy.max_numeric_literal_digits} digits",
            )

    def _validate_symbol_name(self, node: ast.Name) -> None:
        name = node.id
        if name.startswith("_") or "__" in name:
            self.error(node, "forbidden_name", "private names are forbidden")
        if name in ALLOWED_CONSTANTS:
            return
        if name in ALLOWED_FUNCTIONS:
            self.error(
                node,
                "unknown_symbol",
                f"function {name!r} must be called",
            )
        if name not in self.allowed_symbols:
            allowed = ", ".join(sorted(self.allowed_symbols)) or "none"
            self.error(
                node,
                "unknown_symbol",
                f"symbol {name!r} is not allowed in this context; "
                f"allowed symbols: {allowed}",
            )

    @staticmethod
    def _constant_number(node: ast.AST) -> float | None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return None
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _AstSafetyValidator._constant_number(node.operand)
            if value is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            numerator = _AstSafetyValidator._constant_number(node.left)
            denominator = _AstSafetyValidator._constant_number(node.right)
            if numerator is None or denominator in (None, 0.0):
                return None
            return numerator / denominator
        return None


class _AstToSymPy:
    """Construct a SymPy tree directly from an already validated AST."""

    def __init__(self, expression: str, allowed_symbols: Mapping[str, sp.Symbol]):
        self.expression = expression
        self.allowed_symbols = allowed_symbols

    def convert(self, node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Constant):
            source = ast.get_source_segment(self.expression, node) or repr(node.value)
            return sp.Integer(source) if isinstance(node.value, int) else sp.Float(source)
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_CONSTANTS:
                return ALLOWED_CONSTANTS[node.id]
            return self.allowed_symbols[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = self.convert(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left = self.convert(node.left)
            right = self.convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left**right
        if isinstance(node, ast.Call):
            return ALLOWED_FUNCTIONS[node.func.id](
                *(self.convert(argument) for argument in node.args)
            )
        raise AssertionError(f"validated AST contains unexpected {type(node).__name__}")


def _validate_parenthesis_depth(
    expression: str,
    *,
    context: ExpressionContext | str,
    field_index: int | None,
    policy: ExpressionPolicy,
) -> None:
    depth = 0
    for position, character in enumerate(expression, start=1):
        if character in "([{":
            depth += 1
            if depth > policy.max_nesting_depth:
                _raise_expression_error(
                    context=context,
                    expression=expression,
                    error_code="nesting_too_deep",
                    message=f"expression nesting exceeds {policy.max_nesting_depth}",
                    field_index=field_index,
                    position=position,
                )
        elif character in ")]}":
            depth = max(depth - 1, 0)


def parse_restricted_expression(
    expression: str,
    *,
    context: ExpressionContext | str,
    allowed_symbols: Mapping[str, sp.Symbol],
    field_index: int | None = None,
    policy: ExpressionPolicy = DEFAULT_EXPRESSION_POLICY,
) -> sp.Expr:
    """Validate a small math language, then construct its SymPy expression."""
    if not isinstance(expression, str) or not expression.strip():
        _raise_expression_error(
            context=context,
            expression=expression,
            error_code="empty_expression",
            message="expression must be a non-empty string",
            field_index=field_index,
        )
    if len(expression) > policy.max_characters:
        _raise_expression_error(
            context=context,
            expression=expression,
            error_code="expression_too_long",
            message=f"expression exceeds {policy.max_characters} characters",
            field_index=field_index,
        )
    _validate_parenthesis_depth(
        expression,
        context=context,
        field_index=field_index,
        policy=policy,
    )
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError) as exc:
        _raise_expression_error(
            context=context,
            expression=expression,
            error_code="syntax_error",
            message=f"invalid expression syntax: {exc.msg if isinstance(exc, SyntaxError) else exc}",
            field_index=field_index,
            position=getattr(exc, "offset", None),
        )
    validator = _AstSafetyValidator(
        expression,
        context=context,
        allowed_symbols=allowed_symbols,
        field_index=field_index,
        policy=policy,
    )
    validator.validate(tree)
    try:
        return _AstToSymPy(expression, allowed_symbols).convert(tree.body)
    except ExpressionValidationError:
        raise
    except Exception as exc:
        _raise_expression_error(
            context=context,
            expression=expression,
            error_code="sympy_conversion_failed",
            message=f"validated expression could not be constructed: {type(exc).__name__}: {exc}",
            field_index=field_index,
        )


__all__ = [
    "ALLOWED_AST_NODES",
    "ALLOWED_BINARY_OPERATORS",
    "ALLOWED_CONSTANTS",
    "ALLOWED_FUNCTIONS",
    "ALLOWED_UNARY_OPERATORS",
    "DEFAULT_EXPRESSION_POLICY",
    "ExpressionContext",
    "ExpressionPolicy",
    "parse_restricted_expression",
    "validate_identifiers",
]
