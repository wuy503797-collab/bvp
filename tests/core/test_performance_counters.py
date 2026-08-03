"""Deterministic contracts for solver performance counters and timing boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from bvp_core import (
    BVPProblem,
    BVPSolver,
    SolveCancelled,
    SolverConfig,
    SolverCounters,
    solve_bvp_problem,
)
from bvp_core.expressions import SymPyParser


def _exponential() -> BVPProblem:
    return BVPProblem(
        name="Performance exponential",
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
    )


def _two_parameter_identity() -> BVPProblem:
    return BVPProblem(
        name="Two-parameter counter probe",
        odes=["0", "0"],
        var_names=["x", "y"],
        boundary_conditions=["x0_T - 1", "x1_T - 2"],
        known_indices=[],
        unknown_indices=[0, 1],
        known_values={},
        initial_guess=[0.0, 0.0],
    )


def test_counter_snapshot_is_immutable_and_independent() -> None:
    counters = SolverCounters(phi_evaluations=2)
    snapshot = counters.snapshot()
    counters.phi_evaluations += 1

    assert snapshot.phi_evaluations == 2
    with pytest.raises(Exception):
        snapshot.phi_evaluations = 5  # type: ignore[misc]


def test_shooting_counters_match_real_optimizer_and_ivp_calls() -> None:
    result = solve_bvp_problem(
        _exponential(), SolverConfig(method="shooting", eps=1e-8)
    )
    counters = dict(result.solver_metadata["performance_counters"])

    assert counters["root_residual_calls"] == result.solver_metadata["root"]["nfev"]
    assert counters["least_squares_residual_calls"] == 0
    assert counters["phi_evaluations"] == counters["root_residual_calls"]
    assert counters["phi_ivp_solves"] == counters["phi_evaluations"]
    assert counters["ivp_solves"] == (
        counters["phi_ivp_solves"]
        + counters["variational_ivp_solves"]
        + counters["final_validation_ivp_solves"]
    )
    assert counters["final_validation_ivp_solves"] == 1
    assert counters["total_ivp_nfev"] > 0


def test_one_jacobian_uses_n_plus_one_phi_and_no_variational_ivp() -> None:
    solver = BVPSolver(
        _two_parameter_identity(), SolverConfig(method="continuation", eps=1e-8)
    )

    jacobian = solver._dPhi_dp(np.array([0.25, 0.75]))
    counters = solver.counters.snapshot()

    np.testing.assert_allclose(jacobian, np.eye(2), rtol=0.0, atol=1e-8)
    assert counters.jacobian_evaluations == 1
    assert counters.phi_evaluations == 3
    assert counters.phi_ivp_solves == 3
    assert counters.variational_ivp_solves == 0
    assert counters.ivp_solves == 3


def test_jacobian_reuses_only_a_matching_local_phi_base() -> None:
    parameters = np.array([0.25, 0.75])
    reused_solver = BVPSolver(
        _two_parameter_identity(), SolverConfig(method="continuation", eps=1e-8)
    )
    phi_base = reused_solver._Phi(parameters)
    before = reused_solver.counters.phi_evaluations
    reused_solver._set_local_phi_base(parameters, phi_base)
    reused = reused_solver._dPhi_dp(parameters)

    fresh_solver = BVPSolver(
        _two_parameter_identity(), SolverConfig(method="continuation", eps=1e-8)
    )
    fresh = fresh_solver._dPhi_dp(parameters)

    np.testing.assert_array_equal(reused, fresh)
    assert reused_solver.counters.phi_evaluations - before == 2
    assert fresh_solver.counters.phi_evaluations == 3
    reused_solver._set_local_phi_base(parameters, phi_base)
    with pytest.raises(ValueError, match="identical parameter vector"):
        reused_solver._dPhi_dp(parameters + 1.0)


def test_continuation_counters_and_api_core_timing_are_consistent() -> None:
    result = solve_bvp_problem(
        _exponential(),
        SolverConfig(method="continuation", continuation_steps=3, eps=1e-8),
    )
    metadata = result.run_metadata
    counters = dict(result.solver_metadata["performance_counters"])

    assert metadata is not None
    assert metadata.core_elapsed_seconds is not None
    assert metadata.api_elapsed_seconds == metadata.elapsed_seconds
    assert 0 <= metadata.core_elapsed_seconds <= metadata.api_elapsed_seconds
    assert counters["continuation_steps_attempted"] == 3
    assert counters["continuation_steps_completed"] == 3
    assert counters["newton_iterations"] >= counters["newton_updates"]
    assert counters["damping_trials"] >= counters["newton_updates"]
    assert counters["jacobian_evaluations"] > 0
    assert counters["variational_ivp_solves"] == 0
    assert dict(metadata.performance_counters) == counters


def test_numerical_failure_and_cancellation_keep_counter_snapshots() -> None:
    no_root = BVPProblem(
        name="No root performance",
        odes=["0"],
        var_names=["x"],
        boundary_conditions=["x0_T**2 + 1"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )
    failed = solve_bvp_problem(no_root, SolverConfig(method="shooting"))
    failed_counters = dict(failed.run_metadata.performance_counters)
    assert failed.success is False
    assert failed_counters["root_residual_calls"] > 0
    assert failed_counters["least_squares_residual_calls"] > 0

    checks = 0

    def cancel_during_solver() -> None:
        nonlocal checks
        checks += 1
        if checks >= 12:
            raise SolveCancelled("counter cancellation probe")

    with pytest.raises(SolveCancelled) as captured:
        solve_bvp_problem(
            _exponential(),
            SolverConfig(method="continuation", continuation_steps=10),
            cancellation_check=cancel_during_solver,
        )
    cancelled_metadata = captured.value.run_metadata
    assert cancelled_metadata is not None
    assert cancelled_metadata.final_status == "cancelled"
    assert "ivp_solves" in cancelled_metadata.performance_counters
    assert cancelled_metadata.core_elapsed_seconds is not None


def test_one_public_solve_compiles_each_expression_layer_once(monkeypatch) -> None:
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

    monkeypatch.setattr(SymPyParser, "parse", counted_parse)
    monkeypatch.setattr(SymPyParser, "lambdify_all", counted_lambdify)
    monkeypatch.setattr(
        SymPyParser,
        "parse_boundary_conditions",
        staticmethod(counted_boundary),
    )

    result = solve_bvp_problem(_exponential(), SolverConfig(method="shooting"))

    assert result.success is True
    assert calls == {"parse": 1, "lambdify": 1, "boundary": 1}


def test_disabled_solver_event_level_skips_event_construction(monkeypatch) -> None:
    import bvp_core.solver as solver_module

    solver = BVPSolver(_exponential(), SolverConfig(method="shooting"))
    monkeypatch.setattr(solver_module, "solver_event_enabled", lambda _level: False)
    monkeypatch.setattr(
        solver_module,
        "emit_solver_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled event should not be constructed")
        ),
    )

    solver._emit("debug_probe", phase="test", level=10, details={"large": [1]})
