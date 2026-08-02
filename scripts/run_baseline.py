"""Print reproducible phase-one numerical baselines without creating artifacts."""

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
    task_data = json.loads((REPO_ROOT / "task1.json").read_text(encoding="utf-8"))[0]
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
        and ivp_success
        and finite
        and boundary_residual_norm <= BOUNDARY_TOL
    )

    return {
        "case": "26.1 main continuation",
        "method": result["method"],
        "initial_guess": np.asarray(dataset.guess),
        "p_opt": result["p_opt"],
        "solver_reported_success": bool(result["success"]),
        "ivp_success": ivp_success,
        "state_finite": finite,
        "boundary_residual": boundary_residual,
        "boundary_residual_norm": boundary_residual_norm,
        "reported_residual_norm": float(result["residual_norm"]),
        "acceptance_threshold": BOUNDARY_TOL,
        "independent_boundary_validation": boundary_residual_norm <= BOUNDARY_TOL,
        "elapsed_seconds": elapsed,
        "baseline_accepted": accepted,
    }


def run_known_no_root_defect(method: str, defect_id: str) -> dict[str, Any]:
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
        ivp_success and finite and boundary_residual_norm <= dataset.eps
    )
    known_defect = bool(result["success"] and not independent_valid)

    return {
        "case": f"no-real-root {method}",
        "method": method,
        "initial_guess": np.asarray(dataset.guess),
        "p_opt": result["p_opt"],
        "solver_reported_success": bool(result["success"]),
        "ivp_success": ivp_success,
        "state_finite": finite,
        "boundary_residual": boundary_residual,
        "boundary_residual_norm": boundary_residual_norm,
        "acceptance_threshold": dataset.eps,
        "independent_boundary_validation": independent_valid,
        "elapsed_seconds": elapsed,
        "known_defect_id": defect_id,
        "known_defect": known_defect,
        "baseline_accepted": False,
        "status": "KNOWN_DEFECT_REPRODUCED" if known_defect else "DEFECT_NOT_REPRODUCED",
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
    shooting_defect = run_known_no_root_defect("shooting", "BVP-P0-001")
    continuation_defect = run_known_no_root_defect(
        "continuation", "BVP-P0-002"
    )

    for result in (standalone, continuation, shooting_defect, continuation_defect):
        _print_section(result["case"], result)

    positive_baselines_passed = bool(
        standalone["baseline_accepted"] and continuation["baseline_accepted"]
    )
    known_defects_observed = bool(
        shooting_defect["known_defect"] and continuation_defect["known_defect"]
    )
    phase_one_complete = positive_baselines_passed and known_defects_observed
    phase_one_status = (
        "PASS_WITH_KNOWN_DEFECTS" if phase_one_complete else "BASELINE_FAILED"
    )

    _print_section(
        "summary",
        {
            "positive_baselines_passed": positive_baselines_passed,
            "known_defects_observed": known_defects_observed,
            "phase_one_status": phase_one_status,
        },
    )
    return 0 if phase_one_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
