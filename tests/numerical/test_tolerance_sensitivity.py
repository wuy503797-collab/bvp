"""Tolerance sensitivity without claiming mesh convergence or a fitted order."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from main import BVPSolver, Dataset, SymPyParser
from validation_metrics import sample_dense_solution, validate_numerical_solution


def _run_exponential(eps: float) -> dict:
    dataset = Dataset(
        name=f"Exponential tolerance {eps:g}",
        equations=["x"],
        var_names=["x"],
        T=1.0,
        initial_values={0: None},
        boundary_conditions=["x0_T - E"],
        guess=[0.5],
        eps=eps,
        boundary_atol=1e-8,
        method="RK45",
        solver_method="shooting",
        known_indices=[],
        unknown_indices=[0],
    )
    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    solver = BVPSolver(dataset, parser)
    result = solver.solve()
    assert result["success"] is True, {"eps": eps, "result": result}
    t, y = sample_dense_solution(result["sol"].sol, (0.0, 1.0), sample_count=201)
    metrics = validate_numerical_solution(
        t=t,
        numerical_y=y,
        p_opt=result["p_opt"],
        p_exact=np.array([1.0]),
        exact_solution=lambda times: np.exp(times)[np.newaxis, :],
        ode_function=parser.f,
        initial_state_from_parameters=solver._p_to_state,
        boundary_function=solver.bc_residual,
        dense_solution=result["sol"].sol,
    )
    return {"eps": eps, "result": result, "metrics": metrics}


def test_exponential_tolerance_sensitivity_remains_accurate_and_finite() -> None:
    runs = [_run_exponential(eps) for eps in (1e-4, 1e-6, 1e-8)]
    assert all(run["metrics"]["all_finite"] for run in runs)
    assert all(
        run["metrics"]["boundary_residual_norm"] <= 1e-8 for run in runs
    )

    loose_error = runs[0]["metrics"]["max_abs_error"]
    tight_error = runs[-1]["metrics"]["max_abs_error"]
    assert tight_error <= max(10.0 * loose_error, 1e-10), runs
    assert tight_error <= 1e-6, runs[-1]
    assert runs[-1]["metrics"]["parameter_error"] <= 1e-6, runs[-1]
