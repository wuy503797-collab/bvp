"""Compare discrete and differential continuation on the unchanged 26.1 task.

Run: python scripts/run_differential_continuation_validation.py
Evidence is saved outside the frozen report directories.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bvp_core import BVPProblem, SolverConfig  # noqa: E402
from bvp_core.solver import BVPSolver  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "tmp" / "differential_mpp_phase15"


def load_case() -> tuple[BVPProblem, SolverConfig]:
    path = REPO_ROOT / "examples" / "tasks" / "26_1_two_body.json"
    data = json.loads(path.read_text(encoding="utf-8"))[0]
    problem = BVPProblem(
        name=data["name"], odes=data["equations"], var_names=data["var_names"],
        boundary_conditions=data["boundary_conditions"],
        known_indices=data["known_indices"], unknown_indices=data["unknown_indices"],
        known_values={i: data["initial_values"][str(i)] for i in data["known_indices"]},
        initial_guess=data["guess"], t_start=data["t_star"], t_end=data["T"],
    )
    config = SolverConfig(
        method=data["solver_method"], ivp_method=data["method"], eps=data["eps"],
        continuation_steps=data["continuation_steps"],
    )
    return problem, config


def run_case(problem: BVPProblem, config: SolverConfig) -> dict:
    solver = BVPSolver(problem, config)
    started = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - started
    dc = result["solver_metadata"].get("differential_continuation")
    return {
        "method": config.method,
        "p0": list(problem.initial_guess),
        "p_final": result["p_opt"].tolist(),
        "boundary_residual": result["boundary_residual"].tolist(),
        "boundary_residual_norm": float(result["boundary_residual_norm"]),
        "boundary_thresholds": result["boundary_thresholds"].tolist(),
        "boundary_max_scaled_ratio": float(result["boundary_max_scaled_ratio"]),
        "success": bool(result["success"]),
        "status": result["status"], "message": result["message"],
        "ivp_success": bool(result["ivp_success"]),
        "finite_success": bool(result["finite_success"]),
        "boundary_success": bool(result["boundary_success"]),
        "algorithm_success": bool(result["algorithm_success"]),
        "performance_counters": solver.counters.snapshot().to_dict(),
        "elapsed_seconds": elapsed,
        "differential_continuation": dc,
    }


def comparison_passed(discrete: dict, differential: dict) -> bool:
    """A decreasing residual alone must NEVER count as BVP acceptance."""
    for result in (discrete, differential):
        if not all(result.get(key, False) for key in (
            "success", "ivp_success", "finite_success", "boundary_success", "algorithm_success"
        )):
            return False
        residual = np.asarray(result["boundary_residual"], dtype=float)
        thresholds = np.asarray(result["boundary_thresholds"], dtype=float)
        if (not residual.size or residual.shape != thresholds.shape
                or not np.isfinite(residual).all() or not np.isfinite(thresholds).all()
                or not (thresholds > 0).all() or not (np.abs(residual) <= thresholds).all()):
            return False
    dc = differential["differential_continuation"]
    mu = np.asarray(dc["trajectory"]["mu"], dtype=float)
    parameters = np.asarray(dc["trajectory"]["p"], dtype=float)
    return bool(
        dc["mu_solver_success"] and dc["mu_reached_1"]
        and mu.size >= 2 and mu[0] == 0.0 and mu[-1] == 1.0
        and np.isfinite(mu).all() and (np.diff(mu) > 0).all()
        and parameters.shape == (len(differential["p0"]), len(mu))
        and np.isfinite(parameters).all()
        and np.array_equal(parameters[:, 0], differential["p0"])
        and np.allclose(parameters[:, -1], differential["p_final"], rtol=0, atol=1e-14)
    )


def write_plot(trajectory: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for index, row in enumerate(trajectory["p"]):
        ax.plot(trajectory["mu"], row, label=f"p{index + 1}(mu)")
        ax.scatter([trajectory["mu"][0], trajectory["mu"][-1]], [row[0], row[-1]], s=18)
    ax.set(xlabel="mu", ylabel="Unknown initial parameter",
           title="26.1: differential continuation, p(0) to p(1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    problem, config = load_case()
    discrete = run_case(problem, config)
    differential = run_case(problem, replace(config, method="differential_continuation"))
    passed = comparison_passed(discrete, differential)
    evidence = {
        "task": "examples/tasks/26_1_two_body.json",
        "full_initial_state": [2.0, 0.0, *problem.initial_guess],
        "algorithms": {
            "discrete": "p <- p - solve(J(p), Phi(p) - (1-mu_k)*Phi(p0))",
            "differential": "dp/dmu = solve(J(p), -Phi(p0)); p(0)=p0",
        },
        "discrete": discrete, "differential": differential,
        "validation_status": "PASS" if passed else "FAIL",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(json_safe(evidence), indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")
    for label, result in (("discrete", discrete), ("differential", differential)):
        print(f"[{label}]")
        for key, value in result.items():
            if key != "differential_continuation":
                print(f"{key}={value}")
    dc = differential["differential_continuation"]
    print("[mu diagnostics]")
    for key, value in dc.items():
        if key != "trajectory":
            print(f"{key}={value}")
    if dc.get("trajectory", {}).get("mu"):
        plot_path = args.output_dir / "p_mu_26_1.png"
        write_plot(dc["trajectory"], plot_path)
        print(f"p_mu_plot={plot_path}")
    print(f"evidence_json={result_path}")
    print(f"differential_continuation_status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
