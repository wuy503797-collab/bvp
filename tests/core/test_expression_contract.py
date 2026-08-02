"""Behavior contracts for the expression parser before and after migration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bvp_core.expressions import SymPyParser


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parsed(equations: list[str], var_names: list[str]) -> SymPyParser:
    parser = SymPyParser(equations, var_names)
    parser.lambdify_all()
    return parser


def _task(number: str) -> dict:
    tasks = json.loads((REPO_ROOT / "tasks.json").read_text(encoding="utf-8"))
    return next(task for task in tasks if task["name"].startswith(number))


@pytest.mark.parametrize(
    ("equations", "var_names", "t", "state", "expected"),
    [
        (["x**2"], ["x"], 0.25, [3.0], [9.0]),
        (["v", "-x"], ["x", "v"], 0.5, [2.0, -3.0], [-3.0, -2.0]),
        (["6*t"], ["x"], 0.25, [7.0], [1.5]),
        (
            ["sin(t) + cos(x) + sqrt(x**2) + exp(t) + x**3"],
            ["x"],
            0.5,
            [-2.0],
            [np.sin(0.5) + np.cos(-2.0) + 2.0 + np.exp(0.5) - 8.0],
        ),
    ],
)
def test_scalar_time_function_and_multistate_contract(
    equations, var_names, t, state, expected
) -> None:
    values = _parsed(equations, var_names).f(t, np.asarray(state, dtype=float))

    assert values.shape == (len(equations),)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, expected, rtol=1e-13, atol=1e-13)


def test_task_26_1_two_body_expression_order_and_values() -> None:
    task = _task("26.1")
    state = np.array([2.0, 3.0, 4.0, 5.0])
    radius_cubed = (2.0**2 + 3.0**2) ** 1.5
    expected = np.array([4.0, 5.0, -2.0 / radius_cubed, -3.0 / radius_cubed])

    parser = _parsed(task["equations"], task["var_names"])

    np.testing.assert_allclose(parser.f(0.0, state), expected, rtol=1e-14, atol=1e-14)
    assert parser.jac(0.0, state).shape == (4, 4)


def test_task_26_3_smooth_sqrt_and_vectorized_auxiliary_output() -> None:
    task = _task("26.3")
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.5])
    smooth = 0.5 * (
        np.sqrt(1e-10 + (state[5] + 1.0) ** 2)
        - np.sqrt(1e-10 + (state[5] - 1.0) ** 2)
    )
    expected = np.array([2.0, 3.0, smooth, 0.0, -4.0, -5.0])
    parser = _parsed(task["equations"], task["var_names"])

    np.testing.assert_allclose(parser.f(0.0, state), expected, rtol=1e-14, atol=1e-14)

    auxiliary = parser.lambdify_aux(task["aux_outputs"])["u"]
    sample_t = np.linspace(0.0, 1.0, 5)
    sample_y = np.repeat(state[:, np.newaxis], sample_t.size, axis=1)
    sample_y[5] = np.linspace(-2.0, 2.0, sample_t.size)
    values = np.asarray(auxiliary(sample_t, *sample_y), dtype=float)
    expected_aux = 0.5 * (
        np.sqrt(1e-10 + (sample_y[5] + 1.0) ** 2)
        - np.sqrt(1e-10 + (sample_y[5] - 1.0) ** 2)
    )

    assert values.shape == sample_t.shape
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, expected_aux, rtol=1e-14, atol=1e-14)


def test_task_26_4_denominator_expressions_remain_finite_and_ordered() -> None:
    task = _task("26.4")
    state = np.array([4.0, 1.0, -0.54, -0.13, 4.9])
    parser = _parsed(task["equations"], task["var_names"])
    values = parser.f(0.0, state)

    denominator = np.sqrt(state[2] ** 2 + state[3] ** 2)
    control_x = np.sqrt(2.0) * state[2] / denominator
    control_y = np.sqrt(2.0) * state[3] / denominator
    expected = np.array(
        [
            state[4]
            * (
                state[1]
                + control_x
                - 0.5
                * (
                    np.sqrt(1e-6 + (control_x + 1.0) ** 2)
                    - np.sqrt(1e-6 + (control_x - 1.0) ** 2)
                )
            ),
            state[4]
            * (
                -1.5 * state[0]
                - 0.25 * state[1]
                + 0.5
                * (
                    np.sqrt(1e-6 + (control_y + 1.0) ** 2)
                    - np.sqrt(1e-6 + (control_y - 1.0) ** 2)
                )
            ),
            state[4] * 1.5 * state[3],
            state[4] * (-state[2] + 0.25 * state[3]),
            0.0,
        ]
    )

    assert values.shape == (5,)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, expected, rtol=1e-13, atol=1e-13)


def test_boundary_expression_maps_left_and_right_state_in_index_order() -> None:
    residual = SymPyParser.parse_boundary_conditions(
        ["x0_0 + x1_T - 5", "x1_0 - x0_T + sin(pi/2)"],
        ["position", "velocity"],
    )

    values = residual(np.array([2.0, 3.0]), np.array([7.0, 11.0]))

    assert values.shape == (2,)
    np.testing.assert_allclose(values, [8.0, -3.0], rtol=0.0, atol=1e-14)


def test_unknown_symbol_preserves_explicit_parse_failure() -> None:
    with pytest.raises(NameError, match="Symbol"):
        _parsed(["unknown_state + x"], ["x"])


@pytest.mark.parametrize("var_names", [["x", "x"], ["t"]])
def test_state_variable_name_collision_preserves_explicit_failure(var_names) -> None:
    equations = ["x", "-x"] if len(var_names) == 2 else ["t"]
    with pytest.raises(SyntaxError, match="duplicate argument"):
        _parsed(equations, var_names)
