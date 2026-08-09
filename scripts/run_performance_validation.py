"""Reproducible phase-ten before/after solver performance validation."""

from __future__ import annotations

import argparse
import cProfile
import json
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scipy
import sympy
from scipy.integrate import solve_ivp

from bvp_core import (
    BVPProblem,
    BVPSolver,
    SolveRecord,
    SolveRequest,
    SolverConfig,
    build_canonical_export_record,
    canonical_json_text,
    canonical_text_summary,
    solve_bvp_problem,
)
from bvp_core.expressions import SymPyParser


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    problem: BVPProblem
    config: SolverConfig
    repeats: int
    expected_success: bool
    expected_status: str
    expected_parameters: tuple[float, ...] | None = None
    parameter_atol: float = 1e-7


def _exponential() -> BVPProblem:
    return BVPProblem(
        name="Scalar exponential",
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
    )


def _oscillator() -> BVPProblem:
    return BVPProblem(
        name="Harmonic oscillator",
        odes=["v", "-x"],
        var_names=["x", "v"],
        boundary_conditions=["x0_T - 1"],
        known_indices=[0],
        unknown_indices=[1],
        known_values={0: 0.0},
        initial_guess=[0.8],
        t_end=float(np.pi / 2.0),
    )


def _manufactured() -> BVPProblem:
    return BVPProblem(
        name="Manufactured cubic polynomial",
        odes=["v", "6*t"],
        var_names=["y", "v"],
        boundary_conditions=["x0_T"],
        known_indices=[0],
        unknown_indices=[1],
        known_values={0: 1.0},
        initial_guess=[-1.0],
    )


def _two_body(*, guess: tuple[float, float]) -> BVPProblem:
    return BVPProblem(
        name="26.1 Two-body",
        odes=[
            "vx",
            "vy",
            "-x/(x**2 + y**2)**1.5",
            "-y/(x**2 + y**2)**1.5",
        ],
        var_names=["x", "y", "vx", "vy"],
        boundary_conditions=["x0_T - 1.0738644361", "x1_T + 1.0995343576"],
        known_indices=[0, 1],
        unknown_indices=[2, 3],
        known_values={0: 2.0, 1: 0.0},
        initial_guess=guess,
        t_end=7.0,
    )


def _no_root() -> BVPProblem:
    return BVPProblem(
        name="No real root",
        odes=["0"],
        var_names=["x"],
        boundary_conditions=["x0_T**2 + 1"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )


def _singular() -> BVPProblem:
    return BVPProblem(
        name="Singular IVP",
        odes=["1/(t - 0.5)"],
        var_names=["x"],
        boundary_conditions=["x0_T"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )


def _cases() -> tuple[BenchmarkCase, ...]:
    def shooting() -> SolverConfig:
        return SolverConfig(method="shooting", eps=1e-8)

    return (
        BenchmarkCase(
            "exponential", _exponential(), shooting(), 5, True, "success", (1.0,)
        ),
        BenchmarkCase(
            "oscillator", _oscillator(), shooting(), 5, True, "success", (1.0,)
        ),
        BenchmarkCase(
            "manufactured", _manufactured(), shooting(), 5, True, "success", (-2.0,)
        ),
        BenchmarkCase(
            "26.1-shooting",
            _two_body(guess=(-0.5, 0.5)),
            shooting(),
            5,
            True,
            "success",
            (-2.380789650244e-08, 5.000000147112e-01),
            2e-7,
        ),
        BenchmarkCase(
            "26.1-continuation",
            _two_body(guess=(0.5, -0.5)),
            SolverConfig(method="continuation", eps=1e-8, continuation_steps=50),
            3,
            True,
            "success",
            (0.451078122246, -0.299418638352),
            2e-7,
        ),
        BenchmarkCase(
            "no-real-root",
            _no_root(),
            shooting(),
            5,
            False,
            "boundary_residual_too_large",
        ),
        BenchmarkCase(
            "singular-ivp", _singular(), shooting(), 5, False, "ivp_failed"
        ),
    )


_OPTIMIZED_DPHI = BVPSolver._dPhi_dp


def _run_removed_variational_probe(solver: BVPSolver, p: np.ndarray) -> None:
    """Execute only the deleted workload for a same-process legacy comparison."""
    dimension = solver.ds.dim()
    x0 = solver._p_to_state(p)
    identity = np.eye(dimension)
    initial = np.concatenate([x0, identity.flatten()])

    def combined_ode(t, state):
        x = state[:dimension]
        matrix = state[dimension:].reshape(dimension, dimension)
        dxdt = solver.parser.f(t, x)
        dmatrix = solver.parser.jac(t, x) @ matrix
        return np.concatenate([dxdt, dmatrix.flatten()])

    solver.counters.record_ivp_start(purpose="variational")
    solution = solve_ivp(
        combined_ode,
        [solver.ds.t_star, solver.ds.T],
        initial,
        method="RK45",
        rtol=solver.tolerances.jacobian_ivp_rtol,
        atol=solver.tolerances.jacobian_ivp_atol,
        dense_output=False,
    )
    solver.counters.record_ivp_result(solution, purpose="variational")
    if not solution.success:
        raise RuntimeError("legacy variational reference did not complete")
    # Reconstruct the historical XT assignment and deliberately discard it.
    _unused_xt = solution.y[:, -1][dimension:].reshape(dimension, dimension)


def _legacy_dphi_reference(
    solver: BVPSolver,
    p: np.ndarray,
) -> np.ndarray:
    parameters = np.asarray(p, dtype=float)
    solver._local_phi_base = None
    legacy_base = solver._Phi(parameters)
    _run_removed_variational_probe(solver, parameters)
    solver._set_local_phi_base(parameters, legacy_base)
    return _OPTIMIZED_DPHI(solver, parameters)


@contextmanager
def _implementation(*, legacy_reference: bool) -> Iterator[None]:
    if not legacy_reference:
        yield
        return
    current = BVPSolver._dPhi_dp
    BVPSolver._dPhi_dp = _legacy_dphi_reference
    try:
        yield
    finally:
        BVPSolver._dPhi_dp = current


def _measure(
    case: BenchmarkCase,
    *,
    legacy_reference: bool,
    repeats: int | None = None,
) -> dict[str, Any]:
    with _implementation(legacy_reference=legacy_reference):
        solve_bvp_problem(case.problem, case.config)
        core_times: list[float] = []
        api_times: list[float] = []
        results = []
        repeat_count = case.repeats if repeats is None else repeats
        for _ in range(repeat_count):
            result = solve_bvp_problem(case.problem, case.config)
            metadata = result.run_metadata
            assert metadata is not None and metadata.core_elapsed_seconds is not None
            core_times.append(metadata.core_elapsed_seconds)
            api_times.append(metadata.api_elapsed_seconds)
            results.append(result)

    counter_rows = [dict(item.run_metadata.performance_counters) for item in results]
    assert all(row == counter_rows[0] for row in counter_rows[1:])
    result = results[-1]
    counters = counter_rows[0]
    root = dict(result.solver_metadata.get("root") or {})
    least_squares = dict(result.solver_metadata.get("least_squares") or {})
    parameters_ok = (
        True
        if case.expected_parameters is None
        else bool(
            np.allclose(
                result.p_opt,
                case.expected_parameters,
                rtol=0.0,
                atol=case.parameter_atol,
            )
        )
    )
    numerical_ok = bool(
        result.success is case.expected_success
        and result.status == case.expected_status
        and parameters_ok
    )
    counter_ok = bool(
        counters["phi_evaluations"] == counters["phi_ivp_solves"]
        and counters["ivp_solves"]
        == counters["phi_ivp_solves"]
        + counters["variational_ivp_solves"]
        + counters["final_validation_ivp_solves"]
        and counters["total_ivp_nfev"] >= counters["variational_ivp_nfev"]
        and counters["total_ivp_nfev"] >= counters["final_validation_ivp_nfev"]
    )
    return {
        "case": case.name,
        "method": case.config.method,
        "core_median": statistics.median(core_times),
        "core_min": min(core_times),
        "core_max": max(core_times),
        "api_median": statistics.median(api_times),
        "counters": counters,
        "root_nfev": root.get("nfev"),
        "least_squares_nfev": least_squares.get("nfev"),
        "boundary_residual_norm": result.boundary_residual_norm,
        "result": result,
        "numerical_ok": numerical_ok,
        "counter_ok": counter_ok,
    }


def _results_equivalent(before: Any, after: Any) -> bool:
    return bool(
        before.success == after.success
        and before.status == after.status
        and before.message == after.message
        and before.method == after.method
        and np.allclose(before.p_opt, after.p_opt, rtol=0.0, atol=1e-12)
        and np.allclose(before.t, after.t, rtol=0.0, atol=1e-12)
        and np.allclose(before.y, after.y, rtol=0.0, atol=1e-12)
        and np.allclose(
            before.boundary_residual,
            after.boundary_residual,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
    )


def _jacobian_equivalence() -> tuple[float, dict[str, int], dict[str, int]]:
    problem = _two_body(guess=(0.5, -0.5))
    config = SolverConfig(method="continuation", eps=1e-8, continuation_steps=50)
    parameters = np.asarray(problem.initial_guess, dtype=float)
    before_solver = BVPSolver(problem, config)
    with _implementation(legacy_reference=True):
        before = before_solver._dPhi_dp(parameters)
    after_solver = BVPSolver(problem, config)
    after = after_solver._dPhi_dp(parameters)
    return (
        float(np.max(np.abs(before - after))),
        before_solver.counters.snapshot().to_dict(),
        after_solver.counters.snapshot().to_dict(),
    )


def _expression_compilation_counts() -> dict[str, int]:
    calls = {"parse": 0, "lambdify": 0, "boundary": 0}
    original_parse = SymPyParser.parse
    original_lambdify = SymPyParser.lambdify_all
    original_boundary = SymPyParser.parse_boundary_conditions

    def counted_parse(self):
        calls["parse"] += 1
        return original_parse(self)

    def counted_lambdify(self):
        calls["lambdify"] += 1
        return original_lambdify(self)

    def counted_boundary(boundary_conditions, var_names):
        calls["boundary"] += 1
        return original_boundary(boundary_conditions, var_names)

    SymPyParser.parse = counted_parse
    SymPyParser.lambdify_all = counted_lambdify
    SymPyParser.parse_boundary_conditions = staticmethod(counted_boundary)
    try:
        solve_bvp_problem(_exponential(), SolverConfig(method="shooting"))
    finally:
        SymPyParser.parse = original_parse
        SymPyParser.lambdify_all = original_lambdify
        SymPyParser.parse_boundary_conditions = staticmethod(original_boundary)
    return calls


def _metadata_export_valid(row: dict[str, Any], case: BenchmarkCase) -> bool:
    request = SolveRequest.create(
        problem=case.problem,
        config=case.config,
        source_task_id="performance-validation",
    )
    record = SolveRecord(request=request, result=row["result"])
    canonical = build_canonical_export_record(record)
    json_data = json.loads(canonical_json_text(canonical))
    text = canonical_text_summary(canonical)
    counters = json_data["run"].get("performance_counters")
    return bool(
        isinstance(counters, dict)
        and counters.get("ivp_solves", 0) > 0
        and json_data["run"].get("core_elapsed_seconds", -1) >= 0
        and json_data["run"].get("api_elapsed_seconds", -1)
        >= json_data["run"].get("core_elapsed_seconds", 0)
        and "performance_counters:" in text
        and "core_elapsed_seconds:" in text
        and "api_elapsed_seconds:" in text
    )


def _profile_hotspots(
    case: BenchmarkCase, *, legacy_reference: bool
) -> dict[str, float]:
    profiler = cProfile.Profile()
    with _implementation(legacy_reference=legacy_reference):
        profiler.enable()
        solve_bvp_problem(case.problem, case.config)
        profiler.disable()
    selected = {
        "solve_ivp",
        "_Phi",
        "_dPhi_dp",
        "_legacy_dphi_reference",
        "solve",
        "solve_continuation",
        "lambdify_all",
        "build_run_metadata",
    }
    totals = {name: 0.0 for name in selected}
    for entry in profiler.getstats():
        name = getattr(entry.code, "co_name", "")
        if name in totals:
            totals[name] += float(entry.totaltime)
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate deterministic solver performance evidence."
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help=(
            "run one measured solve per case and skip timing profiles; "
            "all deterministic counter and numerical checks remain enabled"
        ),
    )
    arguments = parser.parse_args(argv)

    print("[environment]")
    print(f"python={platform.python_version()}")
    print(f"numpy={np.__version__}")
    print(f"scipy={scipy.__version__}")
    print(f"sympy={sympy.__version__}")
    print(f"system={platform.system()}")
    print(f"machine={platform.machine()}")
    print("scope=current machine and dependency combination only")
    print(f"ci_mode={arguments.ci}")

    cases = _cases()
    repeats = 1 if arguments.ci else None
    before_rows = [
        _measure(case, legacy_reference=True, repeats=repeats) for case in cases
    ]
    after_rows = [
        _measure(case, legacy_reference=False, repeats=repeats) for case in cases
    ]

    print("\n[before-after]")
    print(
        "case method before_median after_median before_ivp after_ivp "
        "before_nfev after_nfev before_phi after_phi jac residual"
    )
    for before, after in zip(before_rows, after_rows):
        before_counters = before["counters"]
        after_counters = after["counters"]
        print(
            before["case"],
            before["method"],
            f"{before['core_median']:.12g}",
            f"{after['core_median']:.12g}",
            before_counters["ivp_solves"],
            after_counters["ivp_solves"],
            before_counters["total_ivp_nfev"],
            after_counters["total_ivp_nfev"],
            before_counters["phi_evaluations"],
            after_counters["phi_evaluations"],
            after_counters["jacobian_evaluations"],
            f"{after['boundary_residual_norm']:.12g}",
        )

    jacobian_difference, jacobian_before, jacobian_after = _jacobian_equivalence()
    print("\n[jacobian]")
    print(f"max_abs_difference={jacobian_difference:.12g}")
    print(f"before_ivp={jacobian_before['ivp_solves']}")
    print(f"after_ivp={jacobian_after['ivp_solves']}")
    print(f"before_variational={jacobian_before['variational_ivp_solves']}")
    print(f"after_variational={jacobian_after['variational_ivp_solves']}")

    print("\n[26.1-continuation-profile]")
    if arguments.ci:
        print("skipped=ci-mode-deterministic-checks-only")
    else:
        before_profile = _profile_hotspots(cases[4], legacy_reference=True)
        after_profile = _profile_hotspots(cases[4], legacy_reference=False)
        for name in sorted(before_profile):
            print(
                f"{name} before={before_profile[name]:.12g} "
                f"after={after_profile[name]:.12g}"
            )

    compilation_counts = _expression_compilation_counts()
    print("\n[expression-compilation]")
    for name, count in compilation_counts.items():
        print(f"{name}={count}")

    counter_ok = all(
        before["counter_ok"] and after["counter_ok"]
        for before, after in zip(before_rows, after_rows)
    )
    numerical_ok = all(
        before["numerical_ok"]
        and after["numerical_ok"]
        and _results_equivalent(before["result"], after["result"])
        for before, after in zip(before_rows, after_rows)
    ) and jacobian_difference <= 1e-12
    redundant_removed = bool(
        any(row["counters"]["variational_ivp_solves"] > 0 for row in before_rows)
        and all(
            row["counters"]["variational_ivp_solves"] == 0 for row in after_rows
        )
    )
    phi_reuse = all(
        (
            before["counters"]["phi_evaluations"]
            - after["counters"]["phi_evaluations"]
            == after["counters"]["jacobian_evaluations"]
            if after["method"] == "continuation"
            else before["counters"]["phi_evaluations"]
            == after["counters"]["phi_evaluations"]
        )
        for before, after in zip(before_rows, after_rows)
    )
    compilation_ok = compilation_counts == {
        "parse": 1,
        "lambdify": 1,
        "boundary": 1,
    }
    export_ok = _metadata_export_valid(after_rows[4], cases[4])
    regression_ok = numerical_ok and all(
        after["counters"]["ivp_solves"] <= before["counters"]["ivp_solves"]
        and after["counters"]["total_ivp_nfev"]
        <= before["counters"]["total_ivp_nfev"]
        for before, after in zip(before_rows, after_rows)
    )
    flags = {
        "counter_consistency_passed": counter_ok,
        "numerical_equivalence_passed": numerical_ok,
        "redundant_variational_work_removed": redundant_removed,
        "phi_reuse_validated": phi_reuse,
        "expression_compilation_count_validated": compilation_ok,
        "metadata_export_passed": export_ok,
        "performance_regression_check_passed": regression_ok,
    }
    print("\nperformance_baseline_collected=PASS")
    for name, passed in flags.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    phase_passed = all(flags.values())
    print(f"phase_ten_status={'PASS' if phase_passed else 'FAIL'}")
    return 0 if phase_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
