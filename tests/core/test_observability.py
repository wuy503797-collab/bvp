"""Contracts for run correlation, immutable metadata, and structured events."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from bvp_core import (
    BVPProblem,
    BVPSolver,
    CancellationToken,
    ExpressionValidationError,
    RunContext,
    SolveCancelled,
    SolverConfig,
    emit_solver_event,
    solve_bvp_problem,
    stable_problem_signature,
)
from bvp_core.observability import build_run_metadata


def _problem(*, name: str = "可观测指数") -> BVPProblem:
    return BVPProblem(
        name=name,
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
    )


def _events(caplog) -> list[dict]:
    return [
        record.solver_event
        for record in caplog.records
        if hasattr(record, "solver_event")
    ]


def test_each_public_solve_has_unique_correlated_run_metadata(caplog) -> None:
    caplog.set_level(logging.INFO, logger="bvp_core.events")
    problem = _problem()
    config = SolverConfig(method="shooting", eps=1e-8)

    first = solve_bvp_problem(problem, config)
    second = solve_bvp_problem(problem, config)

    assert first.run_metadata is not None
    assert second.run_metadata is not None
    assert first.run_metadata.run_id != second.run_metadata.run_id
    assert first.run_metadata.problem_signature == stable_problem_signature(problem)
    first_events = [
        event
        for event in _events(caplog)
        if event["run_id"] == first.run_metadata.run_id
    ]
    assert {event["event_name"] for event in first_events} >= {
        "solve_requested",
        "solve_started",
        "expression_validation_started",
        "solver_started",
        "root_completed",
        "final_validation_completed",
        "solve_succeeded",
    }
    assert {event["run_id"] for event in first_events} == {
        first.run_metadata.run_id
    }


def test_request_context_propagates_without_changing_result() -> None:
    problem = _problem()
    config = SolverConfig(method="shooting")
    context = RunContext.create(
        problem,
        config,
        request_id="request-123",
        source_task_id="task-abc",
    )

    result = solve_bvp_problem(problem, config, run_context=context)

    metadata = result.run_metadata
    assert metadata is not None
    assert metadata.run_id == context.run_id
    assert metadata.request_id == "request-123"
    assert metadata.source_task_id == "task-abc"
    assert metadata.effective_tolerances["effective_ivp_rtol"] == 1e-8


def test_direct_legacy_solver_result_retains_run_correlation() -> None:
    problem = _problem()
    config = SolverConfig(method="shooting")
    solver = BVPSolver(problem, config)

    result = solver.solve()

    metadata = result["solver_metadata"]
    assert metadata["run_id"] == solver.run_context.run_id
    assert metadata["request_id"] is None
    assert metadata["problem_signature"] == stable_problem_signature(problem)


def test_problem_signature_is_stable_and_distinguishes_problem_content() -> None:
    first = _problem()
    same = _problem()
    different = _problem(name="不同名称")

    assert stable_problem_signature(first) == stable_problem_signature(same)
    assert stable_problem_signature(first) != stable_problem_signature(different)
    assert len(stable_problem_signature(first)) == 64


def test_run_metadata_is_utc_nonnegative_immutable_and_private() -> None:
    result = solve_bvp_problem(_problem(), SolverConfig(method="shooting"))
    metadata = result.run_metadata
    assert metadata is not None

    for timestamp in (metadata.started_at, metadata.completed_at):
        assert timestamp.endswith("Z")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert metadata.elapsed_seconds >= 0
    assert metadata.python_version
    assert set(metadata.package_versions) == {"numpy", "scipy", "sympy"}
    serialized = json.dumps(metadata.to_dict(), ensure_ascii=False)
    assert os.path.expanduser("~") not in serialized
    assert os.getcwd() not in serialized
    assert "PATH" not in metadata.to_dict()
    with pytest.raises(FrozenInstanceError):
        metadata.final_status = "changed"
    with pytest.raises(TypeError):
        metadata.effective_tolerances["effective_ivp_rtol"] = 1.0


def test_numerical_failure_still_has_run_metadata_and_terminal_event(caplog) -> None:
    caplog.set_level(logging.INFO, logger="bvp_core.events")
    problem = BVPProblem(
        name="No real root",
        odes=["0"],
        var_names=["x"],
        boundary_conditions=["x0_T**2 + 1"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )

    result = solve_bvp_problem(problem, SolverConfig(method="shooting"))

    assert result.success is False
    assert result.run_metadata is not None
    assert result.run_metadata.final_status == result.status
    terminal = [
        event
        for event in _events(caplog)
        if event["run_id"] == result.run_metadata.run_id
        and event["event_name"] == "solve_failed"
    ]
    assert terminal and terminal[-1]["status"] == result.status


def test_cancelled_run_attaches_metadata_and_terminal_event(caplog) -> None:
    caplog.set_level(logging.INFO, logger="bvp_core.events")
    problem = _problem()
    config = SolverConfig(method="shooting")
    token = CancellationToken()
    token.cancel()
    context = RunContext.create(problem, config, request_id="cancel-request")

    with pytest.raises(SolveCancelled) as captured:
        solve_bvp_problem(
            problem,
            config,
            run_context=context,
            cancellation_check=token.raise_if_cancelled,
        )

    assert captured.value.run_metadata is not None
    assert captured.value.run_metadata.run_id == context.run_id
    assert captured.value.run_metadata.final_status == "cancelled"
    assert any(
        event["event_name"] == "solve_cancelled"
        and event["run_id"] == context.run_id
        for event in _events(caplog)
    )


def test_info_logging_excludes_newton_step_noise(caplog) -> None:
    caplog.set_level(logging.INFO, logger="bvp_core.events")
    solve_bvp_problem(
        _problem(), SolverConfig(method="continuation", continuation_steps=3)
    )

    assert "continuation_step_completed" not in {
        event["event_name"] for event in _events(caplog)
    }


def test_expression_event_uses_bounded_preview_not_full_malicious_text(caplog) -> None:
    caplog.set_level(logging.INFO, logger="bvp_core.events")
    malicious = "__import__('os')" + "x" * 600
    problem = BVPProblem(
        name="Unsafe",
        odes=[malicious],
        var_names=["x"],
        boundary_conditions=["x0_T"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )

    with pytest.raises(ExpressionValidationError):
        solve_bvp_problem(problem, SolverConfig(method="shooting"))

    payload = json.dumps(_events(caplog), ensure_ascii=False)
    assert malicious not in payload
    assert len(payload) < 5000


def test_programming_exception_is_logged_and_not_hidden(monkeypatch, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="bvp_core.events")
    monkeypatch.setattr(
        "bvp_core.api.BVPSolver.solve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bug-marker")),
    )

    with pytest.raises(RuntimeError, match="bug-marker") as captured:
        solve_bvp_problem(_problem(), SolverConfig(method="shooting"))

    assert getattr(captured.value, "run_metadata").final_status == "internal_error"
    assert any(
        event["event_name"] == "solve_failed"
        and event["status"] == "internal_error"
        for event in _events(caplog)
    )


def test_importing_core_is_silent_and_creates_no_log_file(tmp_path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import bvp_core"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )

    assert completed.stdout == ""
    assert completed.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_event_details_redact_paths_and_opaque_object_repr(caplog) -> None:
    caplog.set_level(logging.INFO, logger="bvp_core.events")
    problem = _problem()
    config = SolverConfig(method="shooting")
    context = RunContext.create(problem, config)

    event = emit_solver_event(
        context,
        "privacy_probe",
        phase="test",
        details={
            "path": Path.cwd(),
            "message": f"failed in {Path.home()}",
            "opaque": object(),
        },
    )
    payload = json.dumps(event.to_dict(), ensure_ascii=False)

    assert str(Path.cwd()) not in payload
    assert str(Path.home()) not in payload
    assert "0x" not in payload
    assert "<redacted_path>" in payload


def test_run_metadata_redacts_machine_paths_from_solver_diagnostics() -> None:
    problem = _problem()
    config = SolverConfig(method="shooting")
    context = RunContext.create(problem, config)

    metadata = build_run_metadata(
        context,
        config,
        boundary_count=1,
        final_status="internal_error",
        elapsed_seconds=0.0,
        solver_metadata={"root": {"message": f"failed at {Path.home()}"}},
    )
    payload = json.dumps(metadata.to_dict(), ensure_ascii=False)

    assert str(Path.home()) not in payload
    assert "<redacted_path>" in payload
