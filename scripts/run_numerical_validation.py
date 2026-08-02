"""Run reproducible analytic, manufactured, and tolerance BVP validation."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import BVPSolver, Dataset, SymPyParser
from validation_metrics import sample_dense_solution, validate_numerical_solution


def _build_solver(dataset: Dataset) -> BVPSolver:
    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    return BVPSolver(dataset, parser)


def _exponential_dataset(method: str, eps: float = 1e-8) -> Dataset:
    return Dataset(
        name=f"Scalar exponential ({method}, eps={eps:g})",
        equations=["x"],
        var_names=["x"],
        T=1.0,
        initial_values={0: None},
        boundary_conditions=["x0_T - E"],
        guess=[0.5],
        eps=eps,
        boundary_atol=1e-8,
        method="RK45",
        solver_method=method,
        continuation_steps=10,
        known_indices=[],
        unknown_indices=[0],
        t_star=0.0,
    )


def _oscillator_dataset() -> Dataset:
    return Dataset(
        name="Harmonic oscillator",
        equations=["v", "-x"],
        var_names=["x", "v"],
        T=float(np.pi / 2.0),
        initial_values={0: 0.0, 1: None},
        boundary_conditions=["x0_T - 1"],
        guess=[0.8],
        eps=1e-8,
        boundary_atol=1e-8,
        method="RK45",
        solver_method="shooting",
        known_indices=[0],
        unknown_indices=[1],
        t_star=0.0,
    )


def _manufactured_dataset() -> Dataset:
    return Dataset(
        name="Manufactured cubic polynomial",
        equations=["v", "6*t"],
        var_names=["y", "v"],
        T=1.0,
        initial_values={0: 1.0, 1: None},
        boundary_conditions=["x0_T"],
        guess=[-1.0],
        eps=1e-8,
        boundary_atol=1e-8,
        method="RK45",
        solver_method="shooting",
        known_indices=[0],
        unknown_indices=[1],
        t_star=0.0,
    )


def _metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: metadata[key]
        for key in ("optimizer", "fallback_used", "failure_step", "failure_reason")
        if key in metadata
    }
    for key in ("root", "least_squares"):
        if key in metadata:
            summary[key] = metadata[key]
    if "continuation_steps" in metadata:
        summary["continuation_steps"] = metadata["continuation_steps"]
        summary["completed_steps"] = len(metadata.get("steps", []))
    return summary


def _run_case(
    label: str,
    dataset: Dataset,
    exact_solution: Callable[[np.ndarray], np.ndarray],
    p_exact: np.ndarray,
) -> dict[str, Any]:
    errors = dataset.validate()
    if errors:
        return {
            "case": label,
            "success": False,
            "status": "validation_error",
            "message": "; ".join(errors),
        }

    solver = _build_solver(dataset)
    started = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - started
    if not result["success"]:
        return {
            "case": label,
            "configured_eps": dataset.eps,
            "success": False,
            "status": result["status"],
            "message": result["message"],
            "p_opt": result["p_opt"],
            "elapsed_seconds": elapsed,
            "solver_metadata": _metadata_summary(result["solver_metadata"]),
        }

    validation_t, validation_y = sample_dense_solution(
        result["sol"].sol,
        (dataset.t_star, dataset.T),
        sample_count=201,
    )
    metrics = validate_numerical_solution(
        t=validation_t,
        numerical_y=validation_y,
        p_opt=result["p_opt"],
        p_exact=p_exact,
        exact_solution=exact_solution,
        ode_function=solver.parser.f,
        initial_state_from_parameters=solver._p_to_state,
        boundary_function=solver.bc_residual,
        dense_solution=result["sol"].sol,
    )
    return {
        "case": label,
        "method": dataset.solver_method,
        "configured_eps": dataset.eps,
        "success": bool(result["success"]),
        "status": result["status"],
        "p_opt": result["p_opt"],
        "p_exact": p_exact,
        "parameter_error": metrics["parameter_error"],
        "max_abs_error": metrics["max_abs_error"],
        "rms_error": metrics["rms_error"],
        "state_max_abs_error": metrics["state_max_abs_error"],
        "state_rms_error": metrics["state_rms_error"],
        "boundary_residual": metrics["boundary_residual"],
        "boundary_residual_norm": metrics["boundary_residual_norm"],
        "max_ode_defect": metrics["max_ode_defect"],
        "rms_ode_defect": metrics["rms_ode_defect"],
        "max_ode_defect_per_unit_time": metrics[
            "max_ode_defect_per_unit_time"
        ],
        "all_finite": metrics["all_finite"],
        "validation_sample_count": int(validation_t.size),
        "elapsed_seconds": elapsed,
        "solver_metadata": _metadata_summary(result["solver_metadata"]),
        "_validation_y": validation_y,
    }


def _format(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=12, separator=", ")
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _print_case(case: dict[str, Any]) -> None:
    print(f"\n[{case['case']}]")
    for key, value in case.items():
        if key != "case" and not key.startswith("_"):
            print(f"{key}={_format(value)}")


def _case_passes(case: dict[str, Any], *, max_state_error: float) -> bool:
    return bool(
        case.get("success")
        and case.get("all_finite")
        and case["parameter_error"] <= 1e-6
        and case["max_abs_error"] <= max_state_error
        and case["boundary_residual_norm"] <= 1e-8
        and case["max_ode_defect"] <= 1e-8
    )


def main() -> int:
    exponential_exact = lambda t: np.exp(t)[np.newaxis, :]
    oscillator_exact = lambda t: np.vstack((np.sin(t), np.cos(t)))
    manufactured_exact = lambda t: np.vstack(
        (t**3 - 2.0 * t + 1.0, 3.0 * t**2 - 2.0)
    )

    cases = [
        _run_case(
            "analytic exponential shooting",
            _exponential_dataset("shooting"),
            exponential_exact,
            np.array([1.0]),
        ),
        _run_case(
            "analytic exponential continuation",
            _exponential_dataset("continuation"),
            exponential_exact,
            np.array([1.0]),
        ),
        _run_case(
            "analytic harmonic oscillator shooting",
            _oscillator_dataset(),
            oscillator_exact,
            np.array([1.0]),
        ),
        _run_case(
            "manufactured cubic shooting",
            _manufactured_dataset(),
            manufactured_exact,
            np.array([-2.0]),
        ),
    ]
    for case in cases:
        _print_case(case)

    method_parameter_difference = float(
        np.linalg.norm(cases[0]["p_opt"] - cases[1]["p_opt"])
    )
    method_state_difference = float(
        np.max(np.abs(cases[0]["_validation_y"] - cases[1]["_validation_y"]))
    )
    method_comparison_passed = bool(
        method_parameter_difference <= 1e-7
        and method_state_difference <= 2e-6
    )
    print("\n[shooting versus continuation: analytic exponential]")
    print(f"parameter_difference={method_parameter_difference:.12g}")
    print(f"max_state_difference={method_state_difference:.12g}")
    print(f"comparison_passed={method_comparison_passed}")

    print("\n[tolerance sensitivity: analytic exponential shooting]")
    tolerance_runs = [
        _run_case(
            f"eps={eps:g}",
            _exponential_dataset("shooting", eps=eps),
            exponential_exact,
            np.array([1.0]),
        )
        for eps in (1e-4, 1e-6, 1e-8)
    ]
    print(
        "eps success p_opt parameter_error max_abs_error rms_error "
        "boundary_norm max_ode_defect elapsed_seconds"
    )
    for run in tolerance_runs:
        p_text = _format(run.get("p_opt", np.array([])))
        print(
            f"{run.get('configured_eps')} {run.get('success')} {p_text} "
            f"{run.get('parameter_error')} {run.get('max_abs_error')} "
            f"{run.get('rms_error')} {run.get('boundary_residual_norm')} "
            f"{run.get('max_ode_defect')} {run.get('elapsed_seconds')}"
        )
        print(f"solver_metadata={run.get('solver_metadata')}")

    repeat_first = _run_case(
        "repeat 1",
        _exponential_dataset("shooting"),
        exponential_exact,
        np.array([1.0]),
    )
    repeat_second = _run_case(
        "repeat 2",
        _exponential_dataset("shooting"),
        exponential_exact,
        np.array([1.0]),
    )
    repeatable = bool(
        repeat_first.get("success")
        and repeat_second.get("success")
        and np.allclose(
            repeat_first["p_opt"], repeat_second["p_opt"], rtol=0.0, atol=1e-12
        )
        and np.allclose(
            repeat_first["boundary_residual"],
            repeat_second["boundary_residual"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            repeat_first["max_abs_error"],
            repeat_second["max_abs_error"],
            rtol=0.0,
            atol=1e-15,
        )
        and np.isclose(
            repeat_first["max_ode_defect"],
            repeat_second["max_ode_defect"],
            rtol=0.0,
            atol=1e-15,
        )
    )
    print("\n[repeatability: analytic exponential shooting]")
    print(f"repeatable={repeatable}")
    print(f"first_p_opt={_format(repeat_first.get('p_opt'))}")
    print(f"second_p_opt={_format(repeat_second.get('p_opt'))}")
    print(f"first_boundary={_format(repeat_first.get('boundary_residual'))}")
    print(f"second_boundary={_format(repeat_second.get('boundary_residual'))}")
    print(f"first_max_abs_error={repeat_first.get('max_abs_error')}")
    print(f"second_max_abs_error={repeat_second.get('max_abs_error')}")
    print(f"first_max_ode_defect={repeat_first.get('max_ode_defect')}")
    print(f"second_max_ode_defect={repeat_second.get('max_ode_defect')}")

    tolerance_success = all(run.get("success") for run in tolerance_runs)
    tolerance_finite = all(run.get("all_finite") for run in tolerance_runs)
    loose_error = tolerance_runs[0].get("max_abs_error", float("inf"))
    tight_error = tolerance_runs[-1].get("max_abs_error", float("inf"))
    tolerance_reasonable = bool(
        tolerance_success
        and tolerance_finite
        and tight_error <= max(10.0 * loose_error, 1e-10)
        and tight_error <= 1e-6
    )
    case_success = bool(
        _case_passes(cases[0], max_state_error=1e-6)
        and _case_passes(cases[1], max_state_error=1e-6)
        and _case_passes(cases[2], max_state_error=2e-6)
        and _case_passes(cases[3], max_state_error=2e-6)
    )
    phase_three_passed = bool(
        case_success
        and method_comparison_passed
        and tolerance_reasonable
        and repeatable
    )
    print("\n[summary]")
    print(f"analytic_and_manufactured_cases_passed={case_success}")
    print(f"shooting_continuation_comparison_passed={method_comparison_passed}")
    print(f"tolerance_sensitivity_passed={tolerance_reasonable}")
    print(f"repeatability_passed={repeatable}")
    print(f"phase_three_status={'PASS' if phase_three_passed else 'FAILED'}")
    return 0 if phase_three_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
