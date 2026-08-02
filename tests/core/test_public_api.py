"""End-to-end tests for the no-window public core solve API."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bvp_core import (
    BVPProblem,
    BVPResult,
    BVPValidationError,
    SolverConfig,
    solve_bvp_problem,
)
from validation_metrics import sample_dense_solution, validate_numerical_solution


def _exponential_problem() -> BVPProblem:
    return BVPProblem(
        name="Public API exponential",
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


def _validate_exact_result(
    problem: BVPProblem,
    config: SolverConfig,
    result: BVPResult,
    exact_solution,
    p_exact: np.ndarray,
) -> dict:
    assert result.success is True, result
    assert result.ivp_success is True
    assert result.finite_success is True
    assert result.boundary_success is True
    assert result.sol is not None and callable(result.sol.sol)

    # The public API intentionally exposes no parser/mapper. Reconstructing these
    # pure legacy functions here verifies the public result without GUI state.
    from main import BVPSolver, Dataset, SymPyParser
    from bvp_core.adapters import dataset_kwargs

    dataset = Dataset(**dataset_kwargs(problem, config))
    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    solver = BVPSolver(dataset, parser)
    t, y = sample_dense_solution(
        result.sol.sol, (problem.t_start, problem.t_end), sample_count=201
    )
    return validate_numerical_solution(
        t=t,
        numerical_y=y,
        p_opt=result.p_opt,
        p_exact=p_exact,
        exact_solution=exact_solution,
        ode_function=parser.f,
        initial_state_from_parameters=solver._p_to_state,
        boundary_function=solver.bc_residual,
        dense_solution=result.sol.sol,
    )


def test_importing_core_does_not_import_main_or_create_qapplication() -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["QT_QPA_PLATFORM"] = "offscreen"
    code = (
        "import sys; "
        "from PyQt5.QtWidgets import QApplication; "
        "assert QApplication.instance() is None; "
        "import bvp_core; "
        "assert 'main' not in sys.modules; "
        "assert QApplication.instance() is None; "
        "print('core_import_ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "core_import_ok"


def test_public_api_solves_exponential_without_qapplication() -> None:
    problem = _exponential_problem()
    config = SolverConfig(method="shooting", eps=1e-8, boundary_atol=1e-8)
    result = solve_bvp_problem(problem, config)
    metrics = _validate_exact_result(
        problem,
        config,
        result,
        lambda t: np.exp(t)[np.newaxis, :],
        np.array([1.0]),
    )

    assert isinstance(result, BVPResult)
    assert problem == _exponential_problem()
    assert config == SolverConfig(method="shooting", eps=1e-8, boundary_atol=1e-8)
    assert QApplication.instance() is None
    assert metrics["parameter_error"] <= 1e-6, metrics
    assert metrics["max_abs_error"] <= 1e-6, metrics
    assert metrics["boundary_residual_norm"] <= 1e-8, metrics


def test_public_api_solves_manufactured_two_state_problem() -> None:
    problem = BVPProblem(
        name="Public API manufactured cubic",
        odes=["v", "6*t"],
        var_names=["y", "v"],
        boundary_conditions=["x0_T"],
        known_indices=[0],
        unknown_indices=[1],
        known_values={0: 1.0},
        initial_guess=[-1.0],
        t_start=0.0,
        t_end=1.0,
    )
    config = SolverConfig(method="shooting", eps=1e-8, boundary_atol=1e-8)
    result = solve_bvp_problem(problem, config)
    metrics = _validate_exact_result(
        problem,
        config,
        result,
        lambda t: np.vstack((t**3 - 2.0 * t + 1.0, 3.0 * t**2 - 2.0)),
        np.array([-2.0]),
    )

    assert metrics["parameter_error"] <= 1e-6, metrics
    assert metrics["max_abs_error"] <= 2e-6, metrics
    assert metrics["boundary_residual_norm"] <= 1e-8, metrics


def test_public_api_rejects_no_real_root_with_residual_evidence() -> None:
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
    result = solve_bvp_problem(problem, SolverConfig(method="shooting"))

    assert isinstance(result, BVPResult)
    assert result.success is False
    assert result.status == "boundary_residual_too_large"
    assert result.ivp_success is True
    assert result.boundary_success is False
    np.testing.assert_allclose(result.boundary_residual, [1.0], atol=1e-12)
    assert result.boundary_residual_norm == pytest.approx(1.0)


def test_public_api_preserves_singular_ivp_diagnostics() -> None:
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
    result = solve_bvp_problem(problem, SolverConfig(method="shooting"))

    assert result.success is False
    assert result.status == "ivp_failed"
    assert result.ivp_success is False
    assert result.ivp_status == -1
    assert result.ivp_t_final is not None and result.ivp_t_final < 1.0
    assert "step size" in result.ivp_message.lower()


def test_invalid_problem_is_rejected_before_public_api_solve() -> None:
    with pytest.raises(BVPValidationError, match="boundary condition count"):
        BVPProblem(
            name="Dimension mismatch",
            odes=["x"],
            var_names=["x"],
            boundary_conditions=["x0_T", "x0_0"],
            known_indices=[],
            unknown_indices=[0],
            known_values={},
            initial_guess=[0.0],
        )
