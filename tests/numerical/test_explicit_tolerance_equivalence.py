"""Numerical equivalence and independence checks for explicit tolerances."""

from __future__ import annotations

import numpy as np
import pytest

from bvp_core import BVPProblem, SolverConfig, solve_bvp_problem
from bvp_core.expressions import SymPyParser
from bvp_core.solver import BVPSolver
from validation_metrics import sample_dense_solution, validate_numerical_solution


def _problem() -> BVPProblem:
    return BVPProblem(
        name="Explicit tolerance exponential",
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
        t_start=0.0,
        t_end=1.0,
    )


def _explicit(method: str, eps: float) -> SolverConfig:
    return SolverConfig(
        method=method,
        eps=None,
        ivp_rtol=eps,
        ivp_atol=eps / 10.0,
        root_tol=eps,
        least_squares_ftol=eps,
        least_squares_xtol=eps,
        least_squares_gtol=eps,
        continuation_residual_tol=eps,
        jacobian_relative_step=np.sqrt(eps),
        boundary_atol=1e-8,
        boundary_rtol=0.0,
        continuation_steps=10,
    )


@pytest.mark.parametrize("method", ["shooting", "continuation"])
def test_legacy_and_fully_explicit_equivalent_configs_match(method: str) -> None:
    problem = _problem()
    legacy = solve_bvp_problem(
        problem,
        SolverConfig(
            method=method,
            eps=1e-8,
            boundary_atol=1e-8,
            continuation_steps=10,
        ),
    )
    explicit = solve_bvp_problem(problem, _explicit(method, 1e-8))

    assert legacy.success is True
    assert explicit.success is True
    np.testing.assert_array_equal(explicit.p_opt, legacy.p_opt)
    np.testing.assert_array_equal(explicit.t, legacy.t)
    np.testing.assert_array_equal(explicit.y, legacy.y)
    np.testing.assert_array_equal(
        explicit.boundary_residual, legacy.boundary_residual
    )
    assert legacy.solver_metadata["tolerance_mode"] == "legacy"
    assert explicit.solver_metadata["tolerance_mode"] == "explicit"


def test_changing_only_ivp_tolerances_preserves_other_effective_controls() -> None:
    problem = _problem()
    loose = SolverConfig(
        method="shooting",
        eps=1e-8,
        ivp_rtol=1e-4,
        ivp_atol=1e-5,
        boundary_atol=1e-8,
    )
    tight = SolverConfig(
        method="shooting",
        eps=1e-8,
        ivp_rtol=1e-8,
        ivp_atol=1e-9,
        boundary_atol=1e-8,
    )
    loose_result = solve_bvp_problem(problem, loose)
    tight_result = solve_bvp_problem(problem, tight)

    for result in (loose_result, tight_result):
        assert result.success is True
        assert result.boundary_success is True
        assert result.solver_metadata["effective_root_tol"] == 1e-8
        np.testing.assert_allclose(result.boundary_thresholds, [1e-8])
    assert loose_result.solver_metadata["effective_ivp_rtol"] == 1e-4
    assert tight_result.solver_metadata["effective_ivp_rtol"] == 1e-8

    parser = SymPyParser(list(problem.odes), list(problem.var_names))
    parser.lambdify_all()
    solver = BVPSolver(problem, tight, parser)
    for result in (loose_result, tight_result):
        t, y = sample_dense_solution(result.sol.sol, (0.0, 1.0), sample_count=201)
        metrics = validate_numerical_solution(
            t=t,
            numerical_y=y,
            p_opt=result.p_opt,
            p_exact=np.array([1.0]),
            exact_solution=lambda values: np.exp(values)[np.newaxis, :],
            ode_function=parser.f,
            initial_state_from_parameters=solver._p_to_state,
            boundary_function=solver.bc_residual,
            dense_solution=result.sol.sol,
        )
        assert metrics["all_finite"] is True
        assert metrics["boundary_residual_norm"] <= 1e-8
