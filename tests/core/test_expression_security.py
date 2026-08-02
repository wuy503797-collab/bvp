"""Security contracts for the restricted mathematical expression language."""

from __future__ import annotations

from pathlib import Path

import pytest
import sympy as sp

from bvp_core.exceptions import ExpressionValidationError
from bvp_core.expressions import SymPyParser
from bvp_core.expression_policy import (
    DEFAULT_EXPRESSION_POLICY,
    ExpressionContext,
    parse_restricted_expression,
    validate_identifiers,
)


SYMBOLS = {"t": sp.Symbol("t", real=True), "x": sp.Symbol("x", real=True)}


@pytest.mark.parametrize(
    ("expression", "error_code"),
    [
        ('__import__("os")', "forbidden_name"),
        ('open("file")', "unknown_function"),
        ('eval("1+1")', "unknown_function"),
        ('exec("x=1")', "unknown_function"),
        ("globals()", "unknown_function"),
        ("locals()", "unknown_function"),
        ('getattr(x, "real")', "unknown_function"),
        ('setattr(x, "real", 1)', "unknown_function"),
        ("x.__class__", "attribute_access"),
        ("x.real", "attribute_access"),
        ("foo.bar()", "disallowed_call_target"),
        ("x[0]", "subscript_access"),
        ("(lambda x: x)(1)", "disallowed_call_target"),
        ("[a for a in x]", "disallowed_node"),
        ('{"a": 1}', "disallowed_node"),
        ("(1, 2)", "disallowed_node"),
        ('"string"', "invalid_literal"),
        ('b"bytes"', "invalid_literal"),
        ("0x10", "invalid_literal"),
        ("1_000", "invalid_literal"),
        ('f"{x}"', "disallowed_node"),
        ("x if t else 0", "disallowed_node"),
        ("(x := 1)", "disallowed_node"),
        ("x; 1", "syntax_error"),
        ("x\nopen('file')", "syntax_error"),
        ("x ^ 2", "disallowed_operator"),
        ("2x", "syntax_error"),
    ],
)
def test_dangerous_or_unsupported_syntax_is_rejected_before_conversion(
    expression: str, error_code: str, monkeypatch
) -> None:
    conversion_reached = False

    def unexpected_conversion(*_args, **_kwargs):
        nonlocal conversion_reached
        conversion_reached = True
        raise AssertionError("SymPy conversion must not run for a rejected AST")

    monkeypatch.setattr(
        "bvp_core.expression_policy._AstToSymPy.convert", unexpected_conversion
    )

    with pytest.raises(ExpressionValidationError) as captured:
        parse_restricted_expression(
            expression,
            context=ExpressionContext.ODE,
            allowed_symbols=SYMBOLS,
            field_index=2,
        )

    error = captured.value
    assert error.error_code == error_code
    assert error.context == ExpressionContext.ODE.value
    assert error.field_index == 2
    assert error.expression_preview
    assert len(error.expression_preview) <= error.PREVIEW_LIMIT
    assert error.expression == error.expression_preview
    assert error.reason == error.human_message
    assert conversion_reached is False


@pytest.mark.parametrize(
    ("expression", "error_code"),
    [
        ("unknown + x", "unknown_symbol"),
        ("foo(x)", "unknown_function"),
        ("_private + x", "forbidden_name"),
        ("x__class__ + 1", "forbidden_name"),
        ("x**t", "power_exponent_not_constant"),
        ("x**1000", "power_exponent_out_of_range"),
        ("sin(x=1)", "disallowed_call_arguments"),
    ],
)
def test_names_functions_and_power_policy_have_explicit_errors(
    expression: str, error_code: str
) -> None:
    with pytest.raises(ExpressionValidationError) as captured:
        parse_restricted_expression(
            expression,
            context=ExpressionContext.AUXILIARY,
            allowed_symbols=SYMBOLS,
        )

    assert captured.value.error_code == error_code
    assert captured.value.context == ExpressionContext.AUXILIARY.value


@pytest.mark.parametrize(
    ("expression", "error_code"),
    [
        ("x" * (DEFAULT_EXPRESSION_POLICY.max_characters + 1), "expression_too_long"),
        (
            "(" * (DEFAULT_EXPRESSION_POLICY.max_nesting_depth + 1)
            + "x"
            + ")" * (DEFAULT_EXPRESSION_POLICY.max_nesting_depth + 1),
            "nesting_too_deep",
        ),
        ("+".join(["x"] * 130), "ast_too_complex"),
        (
            "9" * (DEFAULT_EXPRESSION_POLICY.max_numeric_literal_digits + 1),
            "numeric_literal_too_long",
        ),
        (
            "+".join(
                ["sin(x)"] * (DEFAULT_EXPRESSION_POLICY.max_function_calls + 1)
            ),
            "function_call_limit",
        ),
        (
            "+".join(
                ["x**2"] * (DEFAULT_EXPRESSION_POLICY.max_power_operations + 1)
            ),
            "power_operation_limit",
        ),
    ],
)
def test_expression_complexity_limits_are_enforced(
    expression: str, error_code: str
) -> None:
    with pytest.raises(ExpressionValidationError) as captured:
        parse_restricted_expression(
            expression,
            context=ExpressionContext.ODE,
            allowed_symbols=SYMBOLS,
        )

    assert captured.value.error_code == error_code


def test_dangerous_text_has_no_file_or_process_side_effect(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist.txt"
    expression = f'open({str(marker)!r}, "w")'

    with pytest.raises(ExpressionValidationError, match="open"):
        parse_restricted_expression(
            expression,
            context=ExpressionContext.ODE,
            allowed_symbols=SYMBOLS,
        )

    assert marker.exists() is False


@pytest.mark.parametrize(
    "names",
    [
        [""],
        ["x", "x"],
        ["_x"],
        ["x__value"],
        ["for"],
        ["sin"],
        ["pi"],
        ["E"],
        ["t"],
        ["x0_T"],
        ["has space"],
        ["has.dot"],
        ["path/name"],
    ],
)
def test_invalid_or_reserved_state_names_are_rejected(names: list[str]) -> None:
    with pytest.raises(ExpressionValidationError) as captured:
        validate_identifiers(names, context=ExpressionContext.ODE)

    assert captured.value.error_code in {"invalid_identifier", "name_conflict"}


def test_existing_identifier_styles_remain_allowed() -> None:
    validate_identifiers(
        ["T", "psi1", "psi_2", "vx", "x0"], context=ExpressionContext.ODE
    )


def test_allowed_language_maps_directly_to_expected_sympy_expression() -> None:
    expression = (
        "sin(x) + cos(x) + tan(x) + asin(x/2) + acos(x/2) + atan(x) "
        "+ sinh(x) + cosh(x) + tanh(x) + exp(x) + log(x) + sqrt(x**2) "
        "+ abs(x) + Abs(x) + sign(x) + atan2(x, t) + pi + E + x**(3/2)"
    )
    parsed = parse_restricted_expression(
        expression,
        context=ExpressionContext.ODE,
        allowed_symbols=SYMBOLS,
    )

    assert isinstance(parsed, sp.Expr)
    assert parsed.free_symbols <= set(SYMBOLS.values())


def test_context_symbol_whitelist_is_enforced() -> None:
    boundary_symbols = {
        "x0_0": sp.Symbol("x0_0", real=True),
        "x0_T": sp.Symbol("x0_T", real=True),
    }
    with pytest.raises(ExpressionValidationError) as captured:
        parse_restricted_expression(
            "t + x0_T",
            context=ExpressionContext.BOUNDARY,
            allowed_symbols=boundary_symbols,
        )

    assert captured.value.error_code == "unknown_symbol"
    assert "t" in captured.value.human_message


def test_lambdify_failure_retains_context_and_explicit_error_code(monkeypatch) -> None:
    def fail_lambdify(*_args, **_kwargs):
        raise RuntimeError("synthetic construction failure")

    monkeypatch.setattr("bvp_core.expressions.lambdify", fail_lambdify)
    parser = SymPyParser(["x"], ["x"])

    with pytest.raises(ExpressionValidationError) as captured:
        parser.lambdify_all()

    assert captured.value.error_code == "lambdify_failed"
    assert captured.value.context == ExpressionContext.ODE.value
