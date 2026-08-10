"""Print phase-two positive baselines and expected invalid-problem rejections."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import scipy
import sympy
from PyQt5.QtCore import PYQT_VERSION_STR
from scipy.integrate import solve_ivp

from main import BVPSolver, Dataset, SymPyParser
from solver import solve_example_26_1, two_body_dynamics
from task_asset_paths import EXAMPLE_TASKS_DIR


BOUNDARY_TOL = 1e-8
TWO_BODY_TARGET = np.array([1.0738644361, -1.0995343576])


def _format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=12, separator=", ")
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _print_section(title: str, values: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    for key, value in values.items():
        print(f"{key}={_format_value(value)}")


def _build_solver(dataset: Dataset) -> BVPSolver:
    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    return BVPSolver(dataset, parser)


def run_standalone_two_body() -> dict[str, Any]:
    initial_guess = np.array([-0.5, 0.5])
    start = time.perf_counter()
    t, x, y, vx, vy, info = solve_example_26_1(*initial_guess)
    elapsed = time.perf_counter() - start

    p_opt = np.array([info["vx0"], info["vy0"]])
    returned_arrays = np.vstack([t, x, y, vx, vy])
    finite = bool(np.isfinite(returned_arrays).all() and np.isfinite(p_opt).all())
    boundary_residual = np.array([x[-1], y[-1]]) - TWO_BODY_TARGET
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))

    ivp_check = solve_ivp(
        two_body_dynamics,
        [0.0, 7.0],
        [2.0, 0.0, *p_opt],
        method="RK45",
        t_eval=[0.0, 7.0],
        rtol=1e-8,
        atol=1e-9,
    )
    accepted = bool(
        ivp_check.success and finite and boundary_residual_norm <= BOUNDARY_TOL
    )

    return {
        "case": "26.1 standalone shooting",
        "method": "shooting/hybr",
        "initial_guess": initial_guess,
        "p_opt": p_opt,
        "solver_reported_success": True,
        "success_source": "function returned without exception",
        "ivp_success": bool(ivp_check.success),
        "state_finite": finite,
        "sample_count": int(t.size),
        "boundary_residual": boundary_residual,
        "boundary_residual_norm": boundary_residual_norm,
        "acceptance_threshold": BOUNDARY_TOL,
        "independent_boundary_validation": boundary_residual_norm <= BOUNDARY_TOL,
        "elapsed_seconds": elapsed,
        "baseline_accepted": accepted,
    }


def run_main_continuation() -> dict[str, Any]:
    task_data = json.loads(
        (EXAMPLE_TASKS_DIR / "26_1_two_body.json").read_text(encoding="utf-8")
    )[0]
    dataset = Dataset.from_dict(task_data)
    solver = _build_solver(dataset)

    start = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - start

    initial_state = solver._p_to_state(result["p_opt"])
    boundary_residual = solver.bc_residual(initial_state, result["y"][:, -1])
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))
    finite = bool(
        np.isfinite(result["p_opt"]).all()
        and np.isfinite(result["t"]).all()
        and np.isfinite(result["y"]).all()
        and np.isfinite(boundary_residual).all()
    )
    ivp_success = bool(result["sol"].success)
    accepted = bool(
        result["success"]
        and result["ivp_success"]
        and result["finite_success"]
        and result["boundary_success"]
        and ivp_success
        and finite
        and boundary_residual_norm <= dataset.boundary_atol
    )

    return {
        "case": "26.1 main continuation",
        "method": result["method"],
        "initial_guess": np.asarray(dataset.guess),
        "p_opt": result["p_opt"],
        "solver_reported_success": bool(result["success"]),
        "status": result["status"],
        "ivp_success": ivp_success,
        "state_finite": finite,
        "finite_success": result["finite_success"],
        "boundary_success": result["boundary_success"],
        "boundary_residual": boundary_residual,
        "boundary_residual_norm": boundary_residual_norm,
        "reported_residual_norm": float(result["residual_norm"]),
        "acceptance_threshold": dataset.boundary_atol,
        "independent_boundary_validation": (
            boundary_residual_norm <= dataset.boundary_atol
        ),
        "elapsed_seconds": elapsed,
        "baseline_accepted": accepted,
    }


def run_expected_invalid_problem(method: str, defect_id: str) -> dict[str, Any]:
    dataset = Dataset(
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
    )
    solver = _build_solver(dataset)

    start = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - start

    initial_state = solver._p_to_state(result["p_opt"])
    boundary_residual = solver.bc_residual(initial_state, result["y"][:, -1])
    boundary_residual_norm = float(np.linalg.norm(boundary_residual))
    finite = bool(
        np.isfinite(result["p_opt"]).all()
        and np.isfinite(result["y"]).all()
        and np.isfinite(boundary_residual).all()
    )
    ivp_success = bool(result["sol"].success)
    independent_valid = bool(
        ivp_success and finite and boundary_residual_norm <= dataset.boundary_atol
    )
    correctly_rejected = bool(
        result["success"] is False
        and result["boundary_success"] is False
        and not independent_valid
    )

    return {
        "case": f"no-real-root {method}",
        "method": method,
        "initial_guess": np.asarray(dataset.guess),
        "p_opt": result["p_opt"],
        "solver_reported_success": bool(result["success"]),
        "solver_status": result["status"],
        "solver_message": result["message"],
        "ivp_success": ivp_success,
        "state_finite": finite,
        "finite_success": result["finite_success"],
        "boundary_success": result["boundary_success"],
        "boundary_residual": boundary_residual,
        "boundary_residual_norm": boundary_residual_norm,
        "acceptance_threshold": dataset.boundary_atol,
        "boundary_validation": independent_valid,
        "independent_boundary_validation": independent_valid,
        "elapsed_seconds": elapsed,
        "closed_defect_id": defect_id,
        "correctly_rejected": correctly_rejected,
        "baseline_accepted": correctly_rejected,
        "case_status": (
            "EXPECTED_INVALID_PROBLEM_REJECTED"
            if correctly_rejected
            else "INVALID_PROBLEM_HANDLING_FAILED"
        ),
    }


def main() -> int:
    _print_section(
        "environment",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sympy": sympy.__version__,
            "matplotlib": matplotlib.__version__,
            "PyQt5": PYQT_VERSION_STR,
        },
    )

    standalone = run_standalone_two_body()
    continuation = run_main_continuation()
    shooting_rejection = run_expected_invalid_problem("shooting", "BVP-P0-001")
    continuation_rejection = run_expected_invalid_problem(
        "continuation", "BVP-P0-002"
    )

    for result in (
        standalone,
        continuation,
        shooting_rejection,
        continuation_rejection,
    ):
        _print_section(result["case"], result)

    positive_baselines_passed = bool(
        standalone["baseline_accepted"] and continuation["baseline_accepted"]
    )
    invalid_problems_rejected = bool(
        shooting_rejection["correctly_rejected"]
        and continuation_rejection["correctly_rejected"]
    )
    phase_two_complete = positive_baselines_passed and invalid_problems_rejected
    phase_two_status = (
        "PASS" if phase_two_complete else "PHASE_TWO_BASELINE_FAILED"
    )

    _print_section(
        "summary",
        {
            "positive_baselines_passed": positive_baselines_passed,
            "invalid_problems_rejected": invalid_problems_rejected,
            "phase_two_status": phase_two_status,
        },
    )
    return 0 if phase_two_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
