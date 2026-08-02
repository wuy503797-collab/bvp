"""Behavior contracts for the numerical solver migration."""

from __future__ import annotations

import numpy as np
import pytest

from bvp_core import (
    BVPProblem,
    BVPValidationError,
    CancellationToken,
    SolveCancelled,
    SolverConfig,
)
from bvp_core.exceptions import IVPIntegrationError
from bvp_core.expressions import SymPyParser
from bvp_core.solver import BVPSolver
REQUIRED_RESULT_FIELDS = {
    "success",
    "status",
    "message",
    "method",
    "p_opt",
    "t",
    "y",
    "sol",
    "ivp_success",
    "ivp_status",
    "ivp_message",
    "ivp_t_final",
    "optimizer_success",
    "algorithm_success",
    "finite_success",
    "boundary_success",
    "boundary_residual",
    "boundary_residual_norm",
    "boundary_atol",
    "boundary_rtol",
    "boundary_acceptance",
    "solver_metadata",
    "iterations",
    "residual_norm",
}


def _solver(
    problem: BVPProblem,
    config: SolverConfig,
    token: CancellationToken | None = None,
) -> BVPSolver:
    parser = SymPyParser(list(problem.odes), list(problem.var_names))
    parser.lambdify_all()
    return BVPSolver(
        problem,
        config,
        parser,
        cancellation_check=None if token is None else token.raise_if_cancelled,
    )


def _exponential(method: str) -> tuple[BVPProblem, SolverConfig]:
    problem = BVPProblem(
        name=f"Exponential {method}",
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
        t_end=1.0,
    )
    config = SolverConfig(
        method=method,
        eps=1e-8,
        boundary_atol=1e-8,
        continuation_steps=10,
    )
    return problem, config


@pytest.mark.parametrize("method", ["shooting", "continuation"])
def test_solver_success_preserves_legacy_result_shape_and_metadata(method: str) -> None:
    problem, config = _exponential(method)
    result = _solver(problem, config).solve()

    assert REQUIRED_RESULT_FIELDS <= result.keys()
    assert result["success"] is True
    assert result["status"] == "success"
    assert result["method"] == method
    assert result["p_opt"][0] == pytest.approx(1.0, abs=1e-6)
    assert result["boundary_residual_norm"] <= 1e-8
    assert result["residual_norm"] == result["boundary_residual_norm"]
    assert isinstance(result["solver_metadata"], dict)
    if method == "shooting":
        assert result["solver_metadata"]["optimizer"] == "root/hybr"
    else:
        assert result["solver_metadata"]["continuation_steps"] == 10


def test_solver_rejects_no_real_root_with_final_acceptance_fields() -> None:
    problem = BVPProblem(
        name="No real root",
        odes=["0"],
        var_names=["x"],
        boundary_conditions=["x0_T**2 + 1"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )

    result = _solver(problem, SolverConfig(method="shooting")).solve()

    assert result["success"] is False
    assert result["status"] == "boundary_residual_too_large"
    assert result["optimizer_success"] is True
    assert result["ivp_success"] is True
    assert result["finite_success"] is True
    assert result["boundary_success"] is False
    np.testing.assert_allclose(result["boundary_residual"], [1.0], atol=1e-12)
    assert result["boundary_residual_norm"] == pytest.approx(1.0)


def test_solver_preserves_singular_ivp_diagnostics_and_exception_type() -> None:
    problem = BVPProblem(
        name="Singular IVP",
        odes=["1/(t - 0.5)"],
        var_names=["x"],
        boundary_conditions=["x0_T"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )
    config = SolverConfig(method="shooting")
    solver = _solver(problem, config)

    with pytest.raises(IVPIntegrationError) as captured:
        solver._solve_ivp(
            np.asarray(problem.initial_guess), [problem.t_start, problem.t_end]
        )

    error = captured.value
    assert error.ivp_status == -1
    assert error.t_final is not None and error.t_final < problem.t_end
    assert error.state_finite is True
    np.testing.assert_allclose(error.p, problem.initial_guess)

    result = solver.solve()
    assert result["success"] is False
    assert result["status"] == "ivp_failed"
    assert result["ivp_status"] == -1
    assert result["ivp_t_final"] == pytest.approx(error.t_final)


def test_solver_dimension_error_is_rejected_before_numerical_work() -> None:
    with pytest.raises(BVPValidationError, match="boundary condition count is 2"):
        BVPProblem(
            name="Dimension mismatch",
            odes=["v", "-x"],
            var_names=["x", "v"],
            boundary_conditions=["x0_T", "x1_T"],
            known_indices=[0],
            unknown_indices=[1],
            known_values={0: 1.0},
            initial_guess=[0.0],
        )


def test_core_solver_constructor_requires_public_models() -> None:
    problem, config = _exponential("shooting")
    with pytest.raises(TypeError, match="problem must be"):
        BVPSolver(object(), config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must be"):
        BVPSolver(problem, object())  # type: ignore[arg-type]


def test_precancelled_solver_stops_before_numerical_work() -> None:
    token = CancellationToken()
    token.cancel()
    problem, config = _exponential("shooting")

    with pytest.raises(SolveCancelled, match="cancelled cooperatively"):
        _solver(problem, config, token).solve()


def test_continuation_cancellation_checkpoint_preserves_cancelled_semantics() -> None:
    token = CancellationToken()
    problem, config = _exponential("continuation")
    solver = _solver(problem, config, token)

    def cancel_after_start(_method: str, _percent: int, _message: str) -> None:
        token.cancel()

    with pytest.raises(SolveCancelled, match="cancelled cooperatively"):
        solver.solve(callback=cancel_after_start)
