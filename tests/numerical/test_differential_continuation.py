"""Contracts for Davidenko continuation, independent of discrete Newton steps."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import inspect
from types import SimpleNamespace
import textwrap

import numpy as np
import pytest

from bvp_core import BVPProblem, BVPValidationError, SolveCancelled, SolverConfig, solve_bvp_problem
from bvp_core.exceptions import IVPIntegrationError
from bvp_core.solver import BVPSolver
import bvp_core.solver as solver_module
from scripts.run_differential_continuation_validation import (
    comparison_passed, load_case, main as validation_main, run_case,
)


def identity_problem(dimension=2, initial=None):
    return BVPProblem(
        name="Differential identity", odes=["0"] * dimension,
        var_names=[f"v{i}" for i in range(dimension)],
        boundary_conditions=[f"x{i}_T - {i + 1}" for i in range(dimension)],
        known_indices=[], unknown_indices=list(range(dimension)), known_values={},
        initial_guess=initial if initial is not None else [0.0] * dimension,
    )


def identity_solver(dimension=2, initial=None):
    return BVPSolver(identity_problem(dimension, initial),
                     SolverConfig(method="differential_continuation"))


def test_rhs_shape_fixed_phi_and_linear_system(monkeypatch):
    solver = identity_solver()
    jacobian = np.array([[2.0, 1.0], [1.0, 3.0]])
    fixed = np.array([4.0, -2.0])
    original = fixed.copy()
    monkeypatch.setattr(solver, "_dPhi_dp", lambda p, **kwargs: jacobian)
    monkeypatch.setattr(solver, "_Phi", lambda p: pytest.fail("RHS replaced the fixed vector"))
    for mu, parameters in ((0.0, [0.0, 0.0]), (0.7, [7.0, 8.0])):
        rhs = solver.differential_continuation_rhs(mu, np.array(parameters), fixed)
        assert rhs.shape == (2,)
        np.testing.assert_allclose(jacobian @ rhs, -fixed, atol=1e-14)
    np.testing.assert_array_equal(fixed, original)
    assert solver.counters.differential_rhs_evaluations == 2


def test_rhs_jacobian_uses_current_parameters(monkeypatch):
    solver = identity_solver(1)
    monkeypatch.setattr(solver, "_dPhi_dp", lambda p, **kwargs: np.array([[2 * p[0]]]))
    first = solver.differential_continuation_rhs(0.0, np.array([1.0]), np.array([6.0]))
    second = solver.differential_continuation_rhs(0.5, np.array([2.0]), np.array([6.0]))
    np.testing.assert_allclose(first, [-3.0])
    np.testing.assert_allclose(second, [-1.5])


def test_no_explicit_inverse_or_silent_least_squares():
    for method in (BVPSolver.differential_continuation_rhs,
                   BVPSolver.solve_differential_continuation):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        names = [getattr(node.func, "attr", getattr(node.func, "id", ""))
                 for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert not {"inv", "inverse", "pinv", "lstsq"}.intersection(names)


def test_linear_path_reaches_one_with_correct_initial_state():
    result = identity_solver(3).solve()
    assert result["success"], result["message"]
    dc = result["solver_metadata"]["differential_continuation"]
    assert dc["mu_solver_success"] and dc["mu_reached_1"]
    mu = np.asarray(dc["trajectory"]["mu"])
    p = np.asarray(dc["trajectory"]["p"])
    assert mu[0] == 0.0 and mu[-1] == 1.0
    np.testing.assert_array_equal(p[:, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(p, np.arange(1, 4)[:, None] * mu, atol=1e-8)
    np.testing.assert_allclose(result["p_opt"], [1, 2, 3], atol=1e-8)
    assert dc["jacobian_scheme"] == "central"
    assert dc["rhs_evaluations"] > 0


def test_phi_p0_is_frozen_throughout_integration(monkeypatch):
    solver = identity_solver()
    seen = []
    real_rhs = solver.differential_continuation_rhs

    def recording_rhs(mu, p, fixed, **kwargs):
        seen.append((id(fixed), fixed.copy()))
        return real_rhs(mu, p, fixed, **kwargs)

    monkeypatch.setattr(solver, "differential_continuation_rhs", recording_rhs)
    assert solver.solve()["success"]
    assert len(seen) > 1 and len({item[0] for item in seen}) == 1
    for _, fixed in seen:
        np.testing.assert_array_equal(fixed, [-1.0, -2.0])


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_already_accepted_initial_state_has_parameter_major_trajectory(dimension):
    initial = np.arange(1, dimension + 1, dtype=float)
    result = identity_solver(dimension, initial).solve()
    assert result["success"]
    dc = result["solver_metadata"]["differential_continuation"]
    assert dc["integration_skipped"] and dc["rhs_evaluations"] == 0
    assert np.asarray(dc["trajectory"]["p"]).shape == (dimension, 2)
    np.testing.assert_array_equal(dc["trajectory"]["p"], np.repeat(initial[:, None], 2, axis=1))


def test_final_state_ivp_is_reintegrated_and_false_mu_success_is_rejected(monkeypatch):
    solver = identity_solver()
    real_state_ivp = solver._solve_ivp
    integrations = []

    def recording_ivp(p, *args, **kwargs):
        integrations.append(np.asarray(p).copy())
        return real_state_ivp(p, *args, **kwargs)

    def fake_mu(rhs, interval, p0, **kwargs):
        return SimpleNamespace(success=True, t=np.array([0., 1.]),
                               y=np.array([[0., 0.2], [0., 0.4]]), sol=None,
                               status=0, message="Reached mu=1 but wrong BVP parameters")

    monkeypatch.setattr(solver, "_solve_ivp", recording_ivp)
    monkeypatch.setattr(solver_module, "solve_ivp", fake_mu)
    # _solve_ivp also uses the module's solve_ivp. Keep its original integrator.
    from scipy.integrate import solve_ivp as real_ivp
    def dispatch(fun, span, state, **kwargs):
        return fake_mu(fun, span, state, **kwargs) if fun.__name__ == "mu_rhs" else real_ivp(fun, span, state, **kwargs)
    monkeypatch.setattr(solver_module, "solve_ivp", dispatch)
    result = solver.solve()
    assert result["algorithm_success"]
    assert not result["success"] and not result["boundary_success"]
    assert result["status"] == "boundary_residual_too_large"
    np.testing.assert_array_equal(integrations[-1], [0.2, 0.4])
    np.testing.assert_allclose(result["boundary_residual"], [-0.8, -1.6])
    assert len(integrations) == 2  # initial Phi and fresh final validation


@pytest.mark.parametrize("jacobian", [np.zeros((2, 2)), np.diag([1., 1e-14]),
                                    np.full((2, 2), np.nan), np.ones((1, 2))])
def test_jacobian_failure_is_structured_and_never_success(monkeypatch, jacobian):
    solver = identity_solver()
    monkeypatch.setattr(solver, "_dPhi_dp", lambda p, **kwargs: jacobian)
    result = solver.solve()
    assert not result["success"] and not result["algorithm_success"]
    assert result["status"] == "differential_continuation_failed"
    dc = result["solver_metadata"]["differential_continuation"]
    assert dc["failure_type"] in {"LinAlgError", "ValueError"}
    assert not dc["mu_reached_1"]


def test_inner_ivp_failure_propagates_from_rhs(monkeypatch):
    solver = identity_solver()
    def failed(p, **kwargs):
        raise IVPIntegrationError("Injected inner IVP failure", p=p)
    monkeypatch.setattr(solver, "_dPhi_dp", failed)
    result = solver.solve()
    assert not result["success"] and not result["algorithm_success"]
    assert "Injected inner IVP failure" in result["message"]


def test_initial_ivp_failure_returns_structured_failure(monkeypatch):
    solver = identity_solver()
    def failed(p):
        raise IVPIntegrationError("Initial state IVP failed", p=p)
    monkeypatch.setattr(solver, "_Phi", failed)
    result = solver.solve()
    assert not result["success"] and not result["ivp_success"]
    assert solver.counters.differential_rhs_evaluations == 0


@pytest.mark.parametrize("mu_success", [True, False])
def test_incomplete_mu_path_rejected_even_when_boundary_is_satisfied(monkeypatch, mu_success):
    from scipy.integrate import solve_ivp as real_ivp
    def dispatch(fun, span, state, **kwargs):
        if fun.__name__ == "mu_rhs":
            return SimpleNamespace(success=mu_success, t=np.array([0., 0.5]),
                                   y=np.array([[0., 1.], [0., 2.]]), sol=None,
                                   status=-1, message="Stopped before mu=1")
        return real_ivp(fun, span, state, **kwargs)
    monkeypatch.setattr(solver_module, "solve_ivp", dispatch)
    result = identity_solver().solve()
    assert result["boundary_success"]
    assert not result["success"] and not result["algorithm_success"]
    assert result["status"] == "differential_continuation_failed"
    assert result["solver_metadata"]["differential_continuation"]["mu_t_final"] == 0.5


def test_cancellation_through_public_api_retains_diagnostics(monkeypatch):
    def cancelled(self, mu, p, fixed, **kwargs):
        self.counters.differential_rhs_evaluations += 1
        raise SolveCancelled("stop differential integration")
    monkeypatch.setattr(BVPSolver, "differential_continuation_rhs", cancelled)
    with pytest.raises(SolveCancelled) as captured:
        solve_bvp_problem(identity_problem(), SolverConfig(method="differential_continuation"))
    assert captured.value.run_metadata.final_status == "cancelled"
    assert captured.value.run_metadata.solver_method == "differential_continuation"


@pytest.mark.parametrize("options", [
    {"p0": [1]}, {"p0": [np.nan, 0]}, {"mu_rtol": 0}, {"mu_atol": np.inf},
    {"jacobian_step": -1}, {"max_jacobian_condition": 0.5}, {"mu_method": "invalid"},
])
def test_invalid_differential_options_rejected(options):
    with pytest.raises(BVPValidationError):
        identity_solver().solve_differential_continuation(**options)


def test_textbook_26_1_public_api_passes_original_strict_acceptance():
    problem, config = load_case()
    result = solve_bvp_problem(problem, replace(config, method="differential_continuation"))
    assert result.success, result.message
    assert result.ivp_success and result.algorithm_success and result.finite_success
    assert result.boundary_success
    assert np.all(np.abs(result.boundary_residual) <= result.boundary_thresholds)
    assert result.boundary_atol == 1e-8
    dc = result.solver_metadata["differential_continuation"]
    assert dc["mu_reached_1"]
    assert dc["phi_p1_norm"] == result.boundary_residual_norm
    assert result.run_metadata.solver_method == "differential_continuation"
    assert result.solver_metadata["performance_counters"]["differential_rhs_evaluations"] > 0


def test_validation_rejects_reduced_but_unaccepted_residual():
    problem = identity_problem()
    discrete = run_case(problem, SolverConfig(method="continuation"))
    differential = run_case(problem, SolverConfig(method="differential_continuation"))
    assert comparison_passed(discrete, differential)
    bad = deepcopy(differential)
    bad["boundary_residual"] = [1e-4, 1e-4]
    # Even inconsistent success=True fields must not hide an excessive residual.
    assert not comparison_passed(discrete, bad)
    bad = deepcopy(differential)
    bad["success"] = False
    assert not comparison_passed(discrete, bad)


def test_validation_exit_code_fails_on_rejected_bvp(monkeypatch, tmp_path):
    import scripts.run_differential_continuation_validation as validation
    def rejected(problem, config):
        return {"success": False, "differential_continuation": {}}
    monkeypatch.setattr(validation, "run_case", rejected)
    assert validation_main(["--output-dir", str(tmp_path)]) == 1
    assert '"validation_status": "FAIL"' in (tmp_path / "results.json").read_text()
