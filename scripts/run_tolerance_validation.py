"""Deterministic validation of legacy and explicit tolerance semantics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bvp_core import BVPProblem, SolverConfig, solve_bvp_problem
from bvp_core.expressions import SymPyParser
from bvp_core.solver import BVPSolver
from bvp_core.tolerances import evaluate_boundary_acceptance
from main import Dataset
from validation_metrics import sample_dense_solution, validate_numerical_solution


EPS_VALUES = (1e-4, 1e-6, 1e-8)


def _problem() -> BVPProblem:
    return BVPProblem(
        name="Tolerance validation exponential",
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
        t_start=0.0,
        t_end=1.0,
    )


def _explicit(eps: float) -> SolverConfig:
    return SolverConfig(
        method="shooting",
        eps=None,
        ivp_rtol=eps,
        ivp_atol=eps / 10.0,
        root_tol=eps,
        least_squares_ftol=eps,
        least_squares_xtol=eps,
        least_squares_gtol=eps,
        continuation_residual_tol=eps,
        jacobian_relative_step=np.sqrt(eps),
        boundary_atol=1e-8,
        boundary_rtol=0.0,
    )


def _metrics(problem: BVPProblem, config: SolverConfig, result) -> dict:
    parser = SymPyParser(list(problem.odes), list(problem.var_names))
    parser.lambdify_all()
    solver = BVPSolver(problem, config, parser)
    t, y = sample_dense_solution(result.sol.sol, (0.0, 1.0), sample_count=201)
    return validate_numerical_solution(
        t=t,
        numerical_y=y,
        p_opt=result.p_opt,
        p_exact=np.array([1.0]),
        exact_solution=lambda values: np.exp(values)[np.newaxis, :],
        ode_function=parser.f,
        initial_state_from_parameters=solver._p_to_state,
        boundary_function=solver.bc_residual,
        dense_solution=result.sol.sol,
    )


def main() -> int:
    problem = _problem()
    for eps in EPS_VALUES:
        legacy = SolverConfig(method="shooting", eps=eps, boundary_atol=1e-8)
        assert legacy.ivp_rtol == eps
        assert legacy.ivp_atol == eps / 10.0
        assert legacy.root_tol == eps
        assert legacy.least_squares_ftol == eps
        assert legacy.continuation_residual_tol == eps
    print("legacy_mapping_passed=PASS")

    comparisons = []
    for eps in EPS_VALUES:
        legacy_config = SolverConfig(
            method="shooting", eps=eps, boundary_atol=1e-8
        )
        explicit_config = _explicit(eps)
        legacy_result = solve_bvp_problem(problem, legacy_config)
        explicit_result = solve_bvp_problem(problem, explicit_config)
        np.testing.assert_array_equal(explicit_result.p_opt, legacy_result.p_opt)
        np.testing.assert_array_equal(explicit_result.y, legacy_result.y)
        comparisons.append(float(np.max(np.abs(explicit_result.y - legacy_result.y))))
        print(
            f"explicit_equivalence eps={eps:g} "
            f"p_opt={explicit_result.p_opt.tolist()} max_state_difference="
            f"{comparisons[-1]:.12g}"
        )
    print("explicit_equivalence_passed=PASS")

    independent_runs = []
    for ivp_rtol, ivp_atol in ((1e-4, 1e-5), (1e-8, 1e-9)):
        config = SolverConfig(
            method="shooting",
            eps=1e-8,
            ivp_rtol=ivp_rtol,
            ivp_atol=ivp_atol,
            boundary_atol=1e-8,
        )
        result = solve_bvp_problem(problem, config)
        metrics = _metrics(problem, config, result)
        independent_runs.append((config, result, metrics))
        root_metadata = result.to_dict()["solver_metadata"].get("root", {})
        print(
            f"independent_ivp ivp_rtol={ivp_rtol:g} "
            f"p_opt={result.p_opt.tolist()} max_abs_error="
            f"{metrics['max_abs_error']:.12g} boundary_residual_norm="
            f"{metrics['boundary_residual_norm']:.12g} max_ode_defect="
            f"{metrics['max_ode_defect']:.12g} nfev={root_metadata.get('nfev')}"
        )
    assert all(run[1].success for run in independent_runs)
    assert all(run[0].root_tol == 1e-8 for run in independent_runs)
    print("independent_controls_passed=PASS")

    scaled = evaluate_boundary_acceptance(
        [0.5],
        boundary_atol=1e-8,
        boundary_rtol=1e-3,
        boundary_scales=[1000.0],
    )
    assert scaled.success is True
    assert scaled.max_scaled_ratio <= 1.0
    print("scaled_boundary_acceptance_passed=PASS")

    legacy_dataset = Dataset.from_dict({"eps": 1e-6})
    explicit_dataset = Dataset.from_dict(
        {
            "eps": None,
            "ivp_rtol": 1e-8,
            "ivp_atol": 1e-9,
            "root_tol": 1e-8,
            "least_squares_ftol": 1e-8,
            "least_squares_xtol": 1e-8,
            "least_squares_gtol": 1e-8,
            "continuation_residual_tol": 1e-8,
            "jacobian_relative_step": 1e-4,
            "boundary_scales": [1.0],
        }
    )
    assert legacy_dataset.to_dict()["eps"] == 1e-6
    assert explicit_dataset.to_dict()["eps"] is None
    assert explicit_dataset.to_dict()["boundary_scales"] == [1.0]
    print("json_compatibility_passed=PASS")
    print("phase_eight_status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
