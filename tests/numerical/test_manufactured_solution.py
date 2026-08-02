"""Manufactured polynomial solution for non-autonomous two-state validation."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from main import BVPSolver, Dataset, SymPyParser
from validation_metrics import sample_dense_solution, validate_numerical_solution


def test_manufactured_polynomial_solution_recovers_unknown_initial_slope() -> None:
    dataset = Dataset(
        name="Manufactured cubic polynomial",
        equations=["v", "6*t"],
        var_names=["y", "v"],
        T=1.0,
        initial_values={0: 1.0, 1: None},
        boundary_conditions=["x0_T"],
        guess=[-1.0],
        eps=1e-8,
        boundary_atol=1e-8,
        method="RK45",
        solver_method="shooting",
        known_indices=[0],
        unknown_indices=[1],
        t_star=0.0,
    )
    assert dataset.validate() == []

    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    solver = BVPSolver(dataset, parser)
    result = solver.solve()
    assert result["success"] is True, result

    exact_solution = lambda t: np.vstack(
        (t**3 - 2.0 * t + 1.0, 3.0 * t**2 - 2.0)
    )
    t, y = sample_dense_solution(
        result["sol"].sol,
        (dataset.t_star, dataset.T),
        sample_count=201,
    )
    metrics = validate_numerical_solution(
        t=t,
        numerical_y=y,
        p_opt=result["p_opt"],
        p_exact=np.array([-2.0]),
        exact_solution=exact_solution,
        ode_function=parser.f,
        initial_state_from_parameters=solver._p_to_state,
        boundary_function=solver.bc_residual,
        dense_solution=result["sol"].sol,
    )

    assert result["ivp_success"] is True
    assert result["finite_success"] is True
    assert result["boundary_success"] is True
    assert metrics["parameter_error"] <= 1e-6, metrics
    assert metrics["max_abs_error"] <= 2e-6, metrics
    assert metrics["rms_error"] <= 1e-6, metrics
    assert metrics["boundary_residual_norm"] <= 1e-8, metrics
    assert metrics["max_ode_defect"] <= 1e-8, metrics
    assert metrics["max_ode_defect_per_unit_time"] <= 2e-6, metrics
    assert metrics["all_finite"] is True
    np.testing.assert_allclose(result["p_opt"], [-2.0], rtol=0.0, atol=1e-6)
