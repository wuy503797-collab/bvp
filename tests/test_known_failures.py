"""Strict xfail specifications for defects intentionally left for phase two."""

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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "BVP-P0-001: shooting solver accepts optimizer termination with "
        "invalid boundary residual"
    ),
)
def test_shooting_rejects_problem_without_real_boundary_root() -> None:
    dataset = _no_real_root_dataset("shooting")
    solver = _build_solver(dataset)
    result = solver.solve()
    boundary_residual = _independent_boundary_residual(solver, result)
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))

    if not result["sol"].success:
        raise RuntimeError("The known defect requires a successful IVP integration")
    if not np.isfinite(result["y"]).all() or not np.isfinite(boundary_residual).all():
        raise RuntimeError("The known defect requires finite state and residual data")
    if boundary_residual_norm <= dataset.eps:
        raise RuntimeError("The constructed problem unexpectedly satisfies the boundary")

    if not np.allclose(boundary_residual, [1.0], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"unexpected regression residual: {boundary_residual!r}")
    assert result["success"] is False


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "BVP-P0-002: continuation solver returns success without final "
        "residual acceptance"
    ),
)
def test_continuation_rejects_problem_without_real_boundary_root() -> None:
    dataset = _no_real_root_dataset("continuation")
    solver = _build_solver(dataset)
    result = solver.solve()
    boundary_residual = _independent_boundary_residual(solver, result)
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))

    if not result["sol"].success:
        raise RuntimeError("The known defect requires a successful IVP integration")
    if not np.isfinite(result["y"]).all() or not np.isfinite(boundary_residual).all():
        raise RuntimeError("The known defect requires finite state and residual data")
    if boundary_residual_norm <= dataset.eps:
        raise RuntimeError("The constructed problem unexpectedly satisfies the boundary")

    if not np.allclose(boundary_residual, [1.0], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"unexpected regression residual: {boundary_residual!r}")
    assert result["success"] is False


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "BVP-P1-001: validation does not reject a boundary/unknown dimension mismatch"
    ),
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "BVP-P1-002: singular IVP failure is not returned as a diagnostic "
        "success=False result"
    ),
)
def test_singular_ivp_returns_diagnostic_failure_result() -> None:
    task_path = REPO_ROOT / "task_error_test.json"
    task_data = json.loads(task_path.read_text(encoding="utf-8"))[0]
    dataset = Dataset.from_dict(task_data)
    solver = _build_solver(dataset)

    try:
        result = solver.solve()
    except RuntimeError as exc:
        message = str(exc)
        if "solve_ivp" not in message:
            raise
        result = {"success": None, "message": message, "y": None}

    message = str(result.get("message", ""))
    if not message:
        raise RuntimeError("A singular IVP failure must retain a diagnostic message")

    if result.get("success") is True and result.get("y") is not None:
        if not np.isfinite(result["y"]).all():
            raise RuntimeError("Non-finite state data must never be marked successful")

    assert result.get("success") is False
