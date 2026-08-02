"""Unit tests for independent numerical validation metrics."""

from __future__ import annotations

import numpy as np

from validation_metrics import (
    independent_boundary_residual,
    integral_ode_defect,
    parameter_error,
    sample_dense_solution,
    state_error_metrics,
)


def test_state_and_parameter_error_metrics_have_known_values() -> None:
    exact = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    numerical = exact + np.array([[0.0, 0.1, -0.2], [0.05, 0.0, -0.05]])

    metrics = state_error_metrics(numerical, exact)

    assert np.isclose(metrics["max_abs_error"], 0.2, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(metrics["state_max_abs_error"], [0.2, 0.05])
    np.testing.assert_allclose(
        metrics["state_rms_error"],
        np.sqrt(np.mean((numerical - exact) ** 2, axis=1)),
    )
    assert parameter_error([1.0, -2.0], [1.0, -1.5]) == 0.5


def test_boundary_metric_recomputes_project_style_function() -> None:
    numerical_y = np.array([[1.0, 1.5], [-2.0, 0.25]])
    metrics = independent_boundary_residual(
        np.array([-2.0]),
        numerical_y,
        lambda p: np.array([1.0, p[0]]),
        lambda x0, xT: np.array([xT[0] - 1.5, x0[1] + 2.0]),
    )

    np.testing.assert_array_equal(metrics["boundary_residual"], [0.0, 0.0])
    assert metrics["boundary_residual_norm"] == 0.0


def test_simpson_integral_defect_is_exact_for_manufactured_polynomial() -> None:
    t = np.linspace(0.0, 1.0, 21)

    def exact_solution(times: np.ndarray) -> np.ndarray:
        return np.vstack((times**3 - 2.0 * times + 1.0, 3.0 * times**2 - 2.0))

    def ode(time: float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], 6.0 * time])

    states = exact_solution(t)
    defect = integral_ode_defect(
        t,
        states,
        ode,
        dense_solution=exact_solution,
    )

    assert defect["max_ode_defect"] <= 5e-16
    assert defect["rms_ode_defect"] <= 2e-16
    assert defect["max_ode_defect_per_unit_time"] <= 1e-14


def test_dense_solution_sampling_uses_fixed_validation_grid() -> None:
    t, y = sample_dense_solution(
        lambda times: np.vstack((times, times**2)),
        (0.0, 1.0),
        sample_count=17,
    )

    assert t.shape == (17,)
    assert y.shape == (2, 17)
    np.testing.assert_allclose(y[1], t**2)
