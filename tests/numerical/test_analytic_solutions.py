"""Analytic-solution validation for shooting and continuation BVP paths."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from main import BVPSolver, Dataset, SymPyParser
from validation_metrics import sample_dense_solution, validate_numerical_solution


def _build_solver(dataset: Dataset) -> BVPSolver:
    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    return BVPSolver(dataset, parser)


def _solve_and_validate(
    dataset: Dataset,
    exact_solution,
    p_exact: np.ndarray,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    assert dataset.validate() == []
    solver = _build_solver(dataset)
    result = solver.solve()
    assert result["success"] is True, result
    assert result["ivp_success"] is True
    assert result["finite_success"] is True
    assert result["boundary_success"] is True
    assert callable(result["sol"].sol)

    t, y = sample_dense_solution(
        result["sol"].sol,
        (dataset.t_star, dataset.T),
        sample_count=201,
    )
    metrics = validate_numerical_solution(
        t=t,
        numerical_y=y,
        p_opt=result["p_opt"],
        p_exact=p_exact,
        exact_solution=exact_solution,
        ode_function=solver.parser.f,
        initial_state_from_parameters=solver._p_to_state,
        boundary_function=solver.bc_residual,
        dense_solution=result["sol"].sol,
    )
    return result, metrics, t, y


def _exponential_dataset(method: str, eps: float = 1e-8) -> Dataset:
    return Dataset(
        name=f"Analytic exponential ({method})",
        equations=["x"],
        var_names=["x"],
        T=1.0,
        initial_values={0: None},
        boundary_conditions=["x0_T - E"],
        guess=[0.5],
        eps=eps,
        boundary_atol=1e-8,
        method="RK45",
        solver_method=method,
        continuation_steps=10,
        known_indices=[],
        unknown_indices=[0],
        t_star=0.0,
    )


def test_scalar_exponential_shooting_matches_exact_solution() -> None:
    exact = lambda t: np.exp(t)[np.newaxis, :]
    result, metrics, _, _ = _solve_and_validate(
        _exponential_dataset("shooting"), exact, np.array([1.0])
    )

    assert metrics["parameter_error"] <= 1e-6, metrics
    assert metrics["max_abs_error"] <= 1e-6, metrics
    assert metrics["rms_error"] <= 1e-6, metrics
    assert metrics["boundary_residual_norm"] <= 1e-8, metrics
    assert metrics["max_ode_defect"] <= 1e-8, metrics
    assert metrics["max_ode_defect_per_unit_time"] <= 2e-6, metrics
    assert metrics["all_finite"] is True
    np.testing.assert_allclose(result["p_opt"], [1.0], rtol=0.0, atol=1e-6)


def test_harmonic_oscillator_shooting_matches_sine_and_cosine() -> None:
    dataset = Dataset(
        name="Analytic harmonic oscillator",
        equations=["v", "-x"],
        var_names=["x", "v"],
        T=float(np.pi / 2.0),
        initial_values={0: 0.0, 1: None},
        boundary_conditions=["x0_T - 1"],
        guess=[0.8],
        eps=1e-8,
        boundary_atol=1e-8,
        method="RK45",
        solver_method="shooting",
        known_indices=[0],
        unknown_indices=[1],
        t_star=0.0,
    )
    exact = lambda t: np.vstack((np.sin(t), np.cos(t)))
    result, metrics, _, _ = _solve_and_validate(dataset, exact, np.array([1.0]))

    assert dataset.dim() == 2
    np.testing.assert_allclose(
        _build_solver(dataset)._p_to_state(np.array([1.0])),
        [0.0, 1.0],
        rtol=0.0,
        atol=0.0,
    )
    assert metrics["parameter_error"] <= 1e-6, metrics
    assert metrics["max_abs_error"] <= 2e-6, metrics
    assert metrics["boundary_residual_norm"] <= 1e-8, metrics
    assert metrics["max_ode_defect"] <= 1e-8, metrics
    assert metrics["all_finite"] is True
    np.testing.assert_allclose(result["p_opt"], [1.0], rtol=0.0, atol=1e-6)


def test_exponential_shooting_and_continuation_agree() -> None:
    exact = lambda t: np.exp(t)[np.newaxis, :]
    shooting, shooting_metrics, _, shooting_y = _solve_and_validate(
        _exponential_dataset("shooting"), exact, np.array([1.0])
    )
    continuation, continuation_metrics, _, continuation_y = _solve_and_validate(
        _exponential_dataset("continuation"), exact, np.array([1.0])
    )

    assert shooting_metrics["max_abs_error"] <= 1e-6, shooting_metrics
    assert continuation_metrics["max_abs_error"] <= 1e-6, continuation_metrics
    assert shooting_metrics["boundary_residual_norm"] <= 1e-8
    assert continuation_metrics["boundary_residual_norm"] <= 1e-8
    assert np.linalg.norm(shooting["p_opt"] - continuation["p_opt"]) <= 1e-7
    assert np.max(np.abs(shooting_y - continuation_y)) <= 2e-6


def test_exponential_shooting_is_repeatable() -> None:
    exact = lambda t: np.exp(t)[np.newaxis, :]
    first, first_metrics, _, _ = _solve_and_validate(
        _exponential_dataset("shooting"), exact, np.array([1.0])
    )
    second, second_metrics, _, _ = _solve_and_validate(
        _exponential_dataset("shooting"), exact, np.array([1.0])
    )

    np.testing.assert_allclose(first["p_opt"], second["p_opt"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        first_metrics["boundary_residual"],
        second_metrics["boundary_residual"],
        rtol=0.0,
        atol=1e-12,
    )
    assert first_metrics["max_abs_error"] == second_metrics["max_abs_error"]
    assert first_metrics["max_ode_defect"] == second_metrics["max_ode_defect"]
