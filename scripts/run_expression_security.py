"""Run deterministic acceptance checks for the restricted expression language."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import sympy as sp

sys.dont_write_bytecode = True
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bvp_core.exceptions import ExpressionValidationError
from bvp_core.expression_policy import ExpressionContext, parse_restricted_expression
from bvp_core.expressions import SymPyParser
from main import Dataset


TASK_FILES = (
    "tasks.json",
    "task1.json",
    "tasks2.json",
    "tasks3.json",
    "tasks4.json",
    "task5.json",
    "task_error_test.json",
)
EXPECTED_MODEL_INVALID = {("tasks.json", 4)}


def _check_allowed_expressions() -> None:
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    symbols = {"t": t, "x": x}
    cases = (
        "6*t",
        "-x",
        "1 + 2.5 - 3e-4",
        "sin(t) + cos(x) + sqrt(x**2)",
        "-x / (x**2 + t**2)**(3/2)",
        "exp(x) + log(x) + abs(x) + pi + E",
    )
    for expression in cases:
        parsed = parse_restricted_expression(
            expression,
            context=ExpressionContext.ODE,
            allowed_symbols=symbols,
        )
        if not isinstance(parsed, sp.Expr):
            raise AssertionError(f"allowed expression was not converted: {expression}")


def _check_blocked_expressions() -> None:
    x = sp.Symbol("x", real=True)
    symbols = {"x": x}
    cases = {
        '__import__("os")': "forbidden_name",
        'open("file")': "unknown_function",
        'eval("1 + 1")': "unknown_function",
        "x.__class__": "attribute_access",
        "x[0]": "subscript_access",
        "(lambda value: value)(1)": "disallowed_call_target",
        "[value for value in x]": "disallowed_node",
        '{"key": 1}': "disallowed_node",
        "unknown + x": "unknown_symbol",
        "foo(x)": "unknown_function",
        "x**1000": "power_exponent_out_of_range",
    }
    for expression, expected_code in cases.items():
        try:
            parse_restricted_expression(
                expression,
                context=ExpressionContext.AUXILIARY,
                allowed_symbols=symbols,
            )
        except ExpressionValidationError as exc:
            if exc.error_code != expected_code:
                raise AssertionError(
                    f"{expression!r}: expected {expected_code}, got {exc.error_code}"
                ) from exc
        else:
            raise AssertionError(f"blocked expression was accepted: {expression}")


def _check_task_library() -> None:
    for filename in TASK_FILES:
        raw_tasks = json.loads((REPO_ROOT / filename).read_text(encoding="utf-8"))
        for index, raw_task in enumerate(raw_tasks):
            dataset = Dataset.from_dict(raw_task)
            parser = SymPyParser(dataset.equations, dataset.var_names)
            parser.lambdify_all()
            SymPyParser.parse_boundary_conditions(
                dataset.boundary_conditions, dataset.var_names
            )
            parser.lambdify_aux(dataset.aux_outputs)
            model_errors = dataset.validate()
            key = (filename, index)
            if key in EXPECTED_MODEL_INVALID:
                if not any("boundary condition count" in error for error in model_errors):
                    raise AssertionError(
                        f"{filename} task {index + 1}: expected dimension error"
                    )
                status = "PARSE_PASS_MODEL_INVALID_AS_EXPECTED"
            else:
                if model_errors:
                    raise AssertionError(
                        f"{filename} task {index + 1}: {model_errors}"
                    )
                status = "PASS"
            print(f"task_parse_status={filename}#{index + 1}:{status}")


def main() -> int:
    _check_allowed_expressions()
    print("allowed_expression_cases_passed=PASS")
    _check_blocked_expressions()
    print("blocked_expression_cases_passed=PASS")
    _check_task_library()
    print("task_library_parse_passed=PASS")
    print("phase_seven_status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
