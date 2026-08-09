"""Unit tests for centralized numerical and GUI result acceptance."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from main import BVPSolver, Dataset, SymPyParser, is_result_acceptable


def _build_minimal_solver() -> BVPSolver:
    dataset = Dataset(
        name="Acceptance unit test",
        equations=["0"],
        var_names=["x0"],
        T=1.0,
        initial_values={0: None},
        boundary_conditions=["x0_T"],
        guess=[0.0],
        solver_method="shooting",
        known_indices=[],
        unknown_indices=[0],
    )
    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    return BVPSolver(dataset, parser)


def _fake_ivp(*, success: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        status=0 if success else -1,
        message="completed" if success else "required step size is too small",
        t=np.array([0.0, 1.0 if success else 0.5]),
        y=np.array([[0.0, 0.0]]),
    )


def test_optimizer_success_cannot_override_invalid_boundary_residual() -> None:
    solver = _build_minimal_solver()
    result = solver._build_validated_result(
        p=np.array([0.0]),
        sol=_fake_ivp(),
        boundary_residual=np.array([1.0]),
        optimizer_success=True,
        algorithm_success=True,
        method="shooting",
        iterations=1,
    )

    assert result["optimizer_success"] is True
    assert result["ivp_success"] is True
    assert result["finite_success"] is True
    assert result["boundary_success"] is False
    assert result["success"] is False
    assert result["status"] == "boundary_residual_too_large"
    assert result["boundary_residual_norm"] == 1.0


def test_non_finite_candidate_is_rejected() -> None:
    solver = _build_minimal_solver()
    result = solver._build_validated_result(
        p=np.array([np.nan]),
        sol=_fake_ivp(),
        boundary_residual=np.array([0.0]),
        optimizer_success=True,
        algorithm_success=True,
        method="shooting",
        iterations=1,
    )

    assert result["ivp_success"] is True
    assert result["finite_success"] is False
    assert result["boundary_success"] is False
    assert result["success"] is False
    assert result["status"] == "non_finite_result"


def test_ivp_failure_overrides_outer_optimizer_termination() -> None:
    solver = _build_minimal_solver()
    result = solver._build_validated_result(
        p=np.array([0.0]),
        sol=_fake_ivp(success=False),
        boundary_residual=None,
        optimizer_success=True,
        algorithm_success=True,
        method="shooting",
        iterations=1,
    )

    assert result["optimizer_success"] is True
    assert result["ivp_success"] is False
    assert result["boundary_success"] is False
    assert result["success"] is False
    assert result["status"] == "ivp_failed"
    assert "step size" in result["message"].lower()


def test_gui_acceptance_rejects_finished_but_invalid_result() -> None:
    invalid_result = {
        "success": False,
        "ivp_success": True,
        "finite_success": True,
        "boundary_success": False,
    }
    valid_result = {
        "success": True,
        "ivp_success": True,
        "finite_success": True,
        "boundary_success": True,
    }

    assert is_result_acceptable(invalid_result) is False
    assert is_result_acceptable(valid_result) is True


def test_dataset_validation_checks_guess_indices_and_role_overlap() -> None:
    dataset = Dataset(
        equations=["x1", "-x0"],
        var_names=["x0", "x1"],
        initial_values={0: None, 1: 0.0},
        boundary_conditions=["x0_T"],
        guess=[],
        known_indices=[0, 2],
        unknown_indices=[0],
    )

    message = " ".join(dataset.validate()).lower()
    assert "guess=0" in message
    assert "out of range" in message
    assert "overlap" in message
