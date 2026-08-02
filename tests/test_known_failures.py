"""Regression tests for numerical-correctness defects closed in phase two."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from main import BVPSolver, Dataset, SymPyParser


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_solver(dataset: Dataset) -> BVPSolver:
    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    return BVPSolver(dataset, parser)


def _no_real_root_dataset(method: str) -> Dataset:
    return Dataset(
        name=f"No real root ({method})",
        equations=["0"],
        var_names=["x0"],
        T=1.0,
        initial_values={0: None},
        boundary_conditions=["x0_T**2 + 1"],
        guess=[0.0],
        eps=1e-8,
        method="RK45",
        solver_method=method,
        continuation_steps=5,
        known_indices=[],
        unknown_indices=[0],
        t_star=0.0,
    )


def _independent_boundary_residual(
    solver: BVPSolver, result: dict
) -> np.ndarray:
    initial_state = solver._p_to_state(result["p_opt"])
    return solver.bc_residual(initial_state, result["y"][:, -1])


def test_shooting_rejects_problem_without_real_boundary_root() -> None:
    dataset = _no_real_root_dataset("shooting")
    solver = _build_solver(dataset)
    result = solver.solve()
    boundary_residual = _independent_boundary_residual(solver, result)
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))

    assert result["success"] is False
    assert result["status"] == "boundary_residual_too_large"
    assert result["optimizer_success"] is True
    assert result["ivp_success"] is True
    assert result["finite_success"] is True
    assert result["boundary_success"] is False
    assert result["sol"].success is True
    assert np.isfinite(result["y"]).all()
    assert np.isfinite(boundary_residual).all()
    assert boundary_residual_norm > result["boundary_atol"]
    np.testing.assert_allclose(boundary_residual, [1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        result["boundary_residual"], boundary_residual, rtol=0.0, atol=1e-12
    )
    assert result["boundary_residual_norm"] == pytest.approx(
        boundary_residual_norm, rel=0.0, abs=1e-12
    )


def test_continuation_rejects_problem_without_real_boundary_root() -> None:
    dataset = _no_real_root_dataset("continuation")
    solver = _build_solver(dataset)
    result = solver.solve()
    boundary_residual = _independent_boundary_residual(solver, result)
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))

    assert result["success"] is False
    assert result["status"] == "continuation_failed"
    assert result["optimizer_success"] is False
    assert result["ivp_success"] is True
    assert result["finite_success"] is True
    assert result["boundary_success"] is False
    assert result["sol"].success is True
    assert np.isfinite(result["y"]).all()
    assert np.isfinite(boundary_residual).all()
    assert boundary_residual_norm > result["boundary_atol"]
    np.testing.assert_allclose(boundary_residual, [1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        result["boundary_residual"], boundary_residual, rtol=0.0, atol=1e-12
    )
    assert result["boundary_residual_norm"] == pytest.approx(
        boundary_residual_norm, rel=0.0, abs=1e-12
    )


def test_validation_rejects_boundary_unknown_dimension_mismatch() -> None:
    dataset = Dataset(
        name="Boundary dimension mismatch",
        equations=["x1", "-x0"],
        var_names=["x0", "x1"],
        T=1.0,
        initial_values={0: 1.0, 1: None},
        boundary_conditions=["x0_T - 1", "x1_T"],
        guess=[0.0],
        eps=1e-8,
        method="RK45",
        solver_method="shooting",
        known_indices=[0],
        unknown_indices=[1],
    )

    errors = dataset.validate()
    assert errors, "Validation must reject two boundary residuals for one unknown"

    message = " ".join(errors).lower()
    clear_dimension_terms = (
        ("boundary" in message and "unknown" in message)
        or ("гранич" in message and "неизвест" in message)
        or ("边界" in message and "未知" in message)
    )
    assert clear_dimension_terms, "Validation error must explain the dimension mismatch"
    assert "input validation" in message
    assert "2" in message and "1" in message

    solver = _build_solver(dataset)
    with pytest.raises(ValueError, match="boundary condition count is 2"):
        solver.solve()


def test_singular_ivp_returns_diagnostic_failure_result() -> None:
    task_path = REPO_ROOT / "task_error_test.json"
    task_data = json.loads(task_path.read_text(encoding="utf-8"))[0]
    dataset = Dataset.from_dict(task_data)
    solver = _build_solver(dataset)

    result = solver.solve()
    message = str(result.get("message", ""))
    assert result["success"] is False
    assert result["status"] == "ivp_failed"
    assert result["ivp_success"] is False
    assert result["boundary_success"] is False
    assert result["optimizer_success"] is False
    assert result["ivp_status"] is not None
    assert result["ivp_t_final"] is not None
    assert result["ivp_t_final"] < dataset.T
    assert "ivp" in message.lower() or "solve_ivp" in message.lower()
    assert "status" in message.lower()
    assert result["ivp_message"]
    assert not (result["success"] and not np.isfinite(result["y"]).all())
