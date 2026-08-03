"""End-to-end canonical records for success, failure, singular IVP, and cancel."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bvp_core import (
    BVPProblem,
    CancellationToken,
    RunContext,
    SolveCancelled,
    SolveOutcome,
    SolveOutcomeStatus,
    SolveRecord,
    SolveRequest,
    SolverConfig,
    build_canonical_export_record,
    canonical_json_text,
    canonical_text_summary,
    solve_bvp_problem,
)


def _request(problem: BVPProblem, config: SolverConfig, name: str) -> SolveRequest:
    return SolveRequest.create(
        problem=problem,
        config=config,
        source_task_id=name,
        request_id=f"request-{name}",
    )


def _context(request: SolveRequest) -> RunContext:
    return RunContext.create(
        request.problem,
        request.config,
        request_id=request.request_id,
        source_task_id=request.source_task_id,
    )


def _exponential() -> BVPProblem:
    return BVPProblem(
        name="Export exponential",
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
    )


def test_success_record_has_one_machine_fact_source_and_no_raw_solution() -> None:
    config = SolverConfig(method="shooting")
    request = _request(_exponential(), config, "success")
    result = solve_bvp_problem(
        request.problem, request.config, run_context=_context(request)
    )
    record = SolveRecord(request=request, result=result)

    canonical = build_canonical_export_record(record)
    data = json.loads(canonical_json_text(canonical))
    text = canonical_text_summary(canonical)

    assert data["run"]["request_id"] == request.request_id
    assert data["result"]["success"] is True
    assert "sol" not in data["result"]
    assert "raw_solution" not in canonical_json_text(canonical)
    for value in (
        data["schema_version"],
        data["run"]["run_id"],
        data["problem"]["problem_signature"],
        data["result"]["status"],
    ):
        assert value in text


def test_singular_ivp_failure_is_strictly_serializable() -> None:
    problem = BVPProblem(
        name="Singular IVP",
        odes=["1/(t - 0.5)"],
        var_names=["x"],
        boundary_conditions=["x0_T"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.0],
    )
    config = SolverConfig(method="shooting")
    request = _request(problem, config, "singular")
    result = solve_bvp_problem(problem, config, run_context=_context(request))
    outcome = SolveOutcome(
        request=request,
        status=SolveOutcomeStatus.FAILED,
        result=result,
        message=result.message,
    )

    text = canonical_json_text(build_canonical_export_record(outcome))
    data = json.loads(text)

    assert data["result"]["status"] == "ivp_failed"
    assert data["diagnostics"]["ivp"]["status"] == -1
    assert data["run"]["run_id"] == result.run_metadata.run_id
    assert "NaN" not in text and "Infinity" not in text


def test_cancelled_outcome_preserves_run_metadata_and_cancelled_status() -> None:
    config = SolverConfig(method="shooting")
    request = _request(_exponential(), config, "cancel")
    context = _context(request)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(SolveCancelled) as captured:
        solve_bvp_problem(
            request.problem,
            request.config,
            run_context=context,
            cancellation_check=token.raise_if_cancelled,
        )
    outcome = SolveOutcome(
        request=request,
        status=SolveOutcomeStatus.CANCELLED,
        message=str(captured.value),
        run_metadata=captured.value.run_metadata,
    )
    data = build_canonical_export_record(outcome).to_dict()

    assert data["record_type"] == "cancelled"
    assert data["result"]["status"] == "cancelled"
    assert data["run"]["final_status"] == "cancelled"
    assert data["run"]["run_id"] == context.run_id


def test_default_export_privacy_excludes_machine_specific_paths() -> None:
    config = SolverConfig(method="shooting")
    request = _request(_exponential(), config, "privacy")
    result = solve_bvp_problem(
        request.problem, request.config, run_context=_context(request)
    )
    text = canonical_json_text(
        build_canonical_export_record(SolveRecord(request=request, result=result))
    )

    for forbidden in (Path.home(), Path.cwd(), Path(os.sys.executable)):
        assert str(forbidden) not in text
    assert "QApplication" not in text
    assert "0x" not in text
