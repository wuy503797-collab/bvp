"""Positive numerical baselines for the currently working 26.1 paths."""

from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from main import BVPSolver, Dataset, SymPyParser
from solver import solve_example_26_1
from task_asset_paths import EXAMPLE_TASKS_DIR

TARGET_POSITION = np.array([1.0738644361, -1.0995343576])
BOUNDARY_TOL = 1e-8


def test_standalone_two_body_shooting_matches_terminal_boundary() -> None:
    result = solve_example_26_1()

    assert isinstance(result, tuple)
    assert len(result) == 6

    t, x, y, vx, vy, info = result
    arrays = (t, x, y, vx, vy)

    assert all(isinstance(values, np.ndarray) for values in arrays)
    assert all(values.shape == (200,) for values in arrays)
    assert all(np.isfinite(values).all() for values in arrays)
    np.testing.assert_allclose(t[[0, -1]], [0.0, 7.0], rtol=0.0, atol=1e-14)

    boundary_residual = np.array([x[-1], y[-1]]) - TARGET_POSITION
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))

    np.testing.assert_allclose(
        [x[-1], y[-1]], TARGET_POSITION, rtol=0.0, atol=BOUNDARY_TOL
    )
    assert boundary_residual.shape == (2,)
    assert boundary_residual_norm <= BOUNDARY_TOL

    assert set(info) == {"vx0", "vy0", "iter"}
    assert np.isfinite([info["vx0"], info["vy0"]]).all()
    assert isinstance(info["iter"], (int, np.integer))
    assert info["iter"] > 0


def test_main_continuation_two_body_matches_independent_boundary_check() -> None:
    task_path = EXAMPLE_TASKS_DIR / "26_1_two_body.json"
    task_data = json.loads(task_path.read_text(encoding="utf-8"))[0]
    dataset = Dataset.from_dict(task_data)
    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    solver = BVPSolver(dataset, parser)

    result = solver.solve()

    assert result["success"] is True
    assert result["status"] == "success"
    assert result["method"] == "continuation"
    assert result["sol"].success is True
    assert result["ivp_success"] is True
    assert result["optimizer_success"] is True
    assert result["algorithm_success"] is True
    assert result["finite_success"] is True
    assert result["boundary_success"] is True
    assert result["boundary_atol"] == BOUNDARY_TOL
    assert result["boundary_rtol"] == 0.0
    assert result["y"].shape[0] == dataset.dim()
    assert result["y"].shape[1] == result["t"].size
    assert np.isfinite(result["p_opt"]).all()
    assert np.isfinite(result["t"]).all()
    assert np.isfinite(result["y"]).all()

    initial_state = solver._p_to_state(result["p_opt"])
    boundary_residual = solver.bc_residual(initial_state, result["y"][:, -1])
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))

    assert boundary_residual.shape == (dataset.num_unknown(),)
    assert np.isfinite(boundary_residual).all()
    assert boundary_residual_norm <= BOUNDARY_TOL
    np.testing.assert_allclose(
        result["boundary_residual"], boundary_residual, rtol=1e-6, atol=1e-12
    )
    np.testing.assert_allclose(
        result["boundary_residual_norm"],
        boundary_residual_norm,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["residual_norm"], boundary_residual_norm, rtol=1e-6, atol=1e-12
    )
