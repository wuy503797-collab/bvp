"""Verify each resolved tolerance reaches only its numerical consumer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bvp_core import BVPProblem, SolverConfig
from bvp_core.solver import BVPSolver


def _problem() -> BVPProblem:
    return BVPProblem(
        name="Tolerance mapping",
        odes=["0"],
        var_names=["x"],
        boundary_conditions=["x0_T"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )


def test_primary_ivp_receives_only_explicit_ivp_tolerances(monkeypatch) -> None:
    captured = {}

    def fake_solve_ivp(_fun, t_span, x0, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            status=0,
            message="ok",
            t=np.asarray(t_span, dtype=float),
            y=np.column_stack((x0, x0)),
        )

    monkeypatch.setattr("bvp_core.solver.solve_ivp", fake_solve_ivp)
    solver = BVPSolver(
        _problem(),
        SolverConfig(ivp_rtol=2e-7, ivp_atol=3e-9, root_tol=4e-4),
    )
    solver._solve_ivp(np.array([0.0]), [0.0, 1.0])

    assert captured["rtol"] == 2e-7
    assert captured["atol"] == 3e-9


def test_root_receives_root_tol_without_reusing_ivp_tolerance(monkeypatch) -> None:
    captured = {}

    def fake_root(_residual, p0, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(x=p0, success=True, status=1, message="ok", nfev=1)

    monkeypatch.setattr("bvp_core.solver.root", fake_root)
    solver = BVPSolver(
        _problem(), SolverConfig(root_tol=4e-5, ivp_rtol=2e-3)
    )
    monkeypatch.setattr(
        solver,
        "_validate_final_candidate",
        lambda **_kwargs: {"success": False, "message": "captured"},
    )
    solver.solve_shooting()

    assert captured["tol"] == 4e-5


def test_least_squares_receives_three_independent_tolerances(monkeypatch) -> None:
    captured = {}

    def fake_root(_residual, p0, **_kwargs):
        return SimpleNamespace(x=p0, success=False, status=0, message="fallback", nfev=1)

    def fake_least_squares(_residual, p0, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            x=p0,
            success=True,
            status=1,
            message="ok",
            cost=0.0,
            optimality=0.0,
            nfev=1,
        )

    monkeypatch.setattr("bvp_core.solver.root", fake_root)
    monkeypatch.setattr("scipy.optimize.least_squares", fake_least_squares)
    solver = BVPSolver(
        _problem(),
        SolverConfig(
            least_squares_ftol=2e-5,
            least_squares_xtol=3e-6,
            least_squares_gtol=4e-7,
        ),
    )
    monkeypatch.setattr(
        solver,
        "_validate_final_candidate",
        lambda **_kwargs: {"success": False, "message": "captured"},
    )
    solver.solve_shooting()

    assert captured["ftol"] == 2e-5
    assert captured["xtol"] == 3e-6
    assert captured["gtol"] == 4e-7


def test_continuation_uses_only_its_residual_tolerance(monkeypatch) -> None:
    def run_with(residual_tol: float) -> tuple[int, dict]:
        solver = BVPSolver(
            _problem(),
            SolverConfig(
                method="continuation",
                continuation_steps=1,
                continuation_residual_tol=residual_tol,
                boundary_atol=1e-8,
            ),
        )
        phi_calls = 0
        jacobian_calls = 0

        def fake_phi(_p):
            nonlocal phi_calls
            phi_calls += 1
            return np.array([1.0 if phi_calls == 1 else 5e-5])

        def fake_jacobian(_p):
            nonlocal jacobian_calls
            jacobian_calls += 1
            return np.array([[1.0]])

        monkeypatch.setattr(solver, "_Phi", fake_phi)
        monkeypatch.setattr(solver, "_dPhi_dp", fake_jacobian)
        monkeypatch.setattr(
            solver,
            "_validate_final_candidate",
            lambda **kwargs: {
                "success": kwargs["algorithm_success"],
                "status": kwargs["failure_status"] or "validated",
            },
        )
        result = solver.solve_continuation()
        return jacobian_calls, result

    loose_jacobian_calls, loose_result = run_with(1e-4)
    strict_jacobian_calls, strict_result = run_with(1e-6)

    assert loose_jacobian_calls == 0
    assert loose_result["status"] == "validated"
    assert strict_jacobian_calls > 0
    assert strict_result["success"] is False
    assert strict_result["status"] == "continuation_failed"


def test_boundary_acceptance_does_not_replace_algorithm_success() -> None:
    solver = BVPSolver(
        _problem(),
        SolverConfig(boundary_atol=2.0, boundary_rtol=0.0),
    )
    solution = SimpleNamespace(
        success=True,
        status=0,
        message="ok",
        t=np.array([0.0, 1.0]),
        y=np.array([[0.0, 0.0]]),
    )

    result = solver._build_validated_result(
        p=np.array([0.0]),
        sol=solution,
        boundary_residual=np.array([1.0]),
        optimizer_success=False,
        algorithm_success=False,
        method="shooting",
        iterations=1,
    )

    assert result["boundary_success"] is True
    assert result["algorithm_success"] is False
    assert result["success"] is False


def test_strict_boundary_rejects_optimizer_success() -> None:
    solver = BVPSolver(_problem(), SolverConfig(boundary_atol=1e-8))
    solution = SimpleNamespace(
        success=True,
        status=0,
        message="ok",
        t=np.array([0.0, 1.0]),
        y=np.array([[0.0, 0.0]]),
    )

    result = solver._build_validated_result(
        p=np.array([0.0]),
        sol=solution,
        boundary_residual=np.array([2e-8]),
        optimizer_success=True,
        algorithm_success=True,
        method="shooting",
        iterations=1,
    )

    assert result["optimizer_success"] is True
    assert result["boundary_success"] is False
    assert result["success"] is False
