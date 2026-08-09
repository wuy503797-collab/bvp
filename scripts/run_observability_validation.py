"""Deterministic terminal validation for phase-nine observability and exports."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bvp_core import (
    BVPProblem,
    RunContext,
    SolveOutcome,
    SolveOutcomeStatus,
    SolveRecord,
    SolveRequest,
    SolverConfig,
    build_canonical_export_record,
    solve_bvp_problem,
    write_canonical_json,
    write_canonical_text,
)


class _MemoryHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "solver_event", None)
        if event is not None:
            self.events.append(event)


def _exponential() -> BVPProblem:
    return BVPProblem(
        name="Observability 指数 / Экспонента",
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
        auxiliary_expressions={"double": "2*x"},
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


def main() -> int:
    logger = logging.getLogger("bvp_core.events")
    memory = _MemoryHandler()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(memory)
    try:
        problem = _exponential()
        config = SolverConfig(method="shooting", eps=1e-8)
        request = SolveRequest.create(
            problem=problem,
            config=config,
            source_task_id="observability-task",
            request_id="observability-request",
        )
        context = RunContext.create(
            problem,
            config,
            request_id=request.request_id,
            source_task_id=request.source_task_id,
        )
        result = solve_bvp_problem(problem, config, run_context=context)
        second = solve_bvp_problem(problem, config)

        event_names = {event["event_name"] for event in memory.events}
        assert {
            "solve_requested",
            "solve_started",
            "solver_started",
            "root_completed",
            "final_validation_completed",
            "solve_succeeded",
        } <= event_names
        assert all(event["run_id"] for event in memory.events)
        print("structured_events_passed=PASS")

        metadata = result.run_metadata
        assert metadata is not None
        assert metadata.run_id == context.run_id
        assert metadata.request_id == request.request_id
        assert metadata.run_id != second.run_metadata.run_id
        assert all(
            event["run_id"] == context.run_id
            for event in memory.events
            if event["request_id"] == request.request_id
        )
        print("run_correlation_passed=PASS")

        metadata_data = metadata.to_dict()
        assert metadata_data["schema_version"] == "bvp-run-v1"
        assert metadata_data["elapsed_seconds"] >= 0
        assert metadata_data["started_at"].endswith("Z")
        assert metadata_data["completed_at"].endswith("Z")
        assert set(metadata_data["package_versions"]) == {"numpy", "scipy", "sympy"}
        assert metadata_data["effective_tolerances"]["effective_ivp_rtol"] == 1e-8
        print("metadata_completeness_passed=PASS")

        record = SolveRecord(
            request=request,
            result=result,
            auxiliary_outputs={"double": 2.0 * result.y[0]},
            auxiliary_sample_t=result.t,
            auxiliary_errors=("controlled auxiliary warning",),
        )
        canonical = build_canonical_export_record(record)
        with tempfile.TemporaryDirectory(prefix="bvp-observability-") as directory:
            target = Path(directory)
            json_path = target / "结果.json"
            text_path = target / "摘要.txt"
            write_canonical_json(json_path, canonical)
            json_data = json.loads(json_path.read_text(encoding="utf-8"))
            assert json_data["schema_version"] == "bvp-result-v1"
            assert json_data["run"]["run_id"] == context.run_id
            assert json_data["problem"]["name"] == problem.name
            assert "NaN" not in json_path.read_text(encoding="utf-8")
            print("json_export_passed=PASS")

            write_canonical_text(text_path, canonical)
            text = text_path.read_text(encoding="utf-8")
            assert "schema_version: bvp-result-v1" in text
            assert f"run_id: {context.run_id}" in text
            assert problem.name in text
            assert "boundary_thresholds:" in text
            print("text_export_passed=PASS")

            json_path.write_text("old", encoding="utf-8")
            write_canonical_json(json_path, canonical)
            assert json.loads(json_path.read_text(encoding="utf-8"))["run"][
                "run_id"
            ] == context.run_id
            assert not list(target.glob(".*.tmp"))
            print("atomic_write_passed=PASS")

            failed_result = solve_bvp_problem(
                _no_root(), SolverConfig(method="shooting")
            )
            failed_request = SolveRequest.create(
                problem=_no_root(),
                config=SolverConfig(method="shooting"),
                source_task_id="failure-task",
            )
            failed_outcome = SolveOutcome(
                request=failed_request,
                status=SolveOutcomeStatus.FAILED,
                result=failed_result,
                message=failed_result.message,
            )
            failure_data = build_canonical_export_record(failed_outcome).to_dict()
            assert failure_data["record_type"] == "failure_diagnostic"
            assert failure_data["result"]["success"] is False
            assert failure_data["diagnostics"]["boundary_acceptance"]["success"] is False
            assert failure_data["run"]["final_status"] == failed_result.status
            print("failure_diagnostics_passed=PASS")

            combined = json_path.read_text(encoding="utf-8") + json.dumps(
                memory.events, ensure_ascii=False
            )
            forbidden = {
                str(REPO_ROOT),
                str(Path.home()),
                sys.executable,
                os.environ.get("USERNAME", "__missing_username__"),
            }
            assert all(not value or value not in combined for value in forbidden)
            assert "0x" not in combined
            print("privacy_boundary_passed=PASS")
    finally:
        logger.removeHandler(memory)
        logger.setLevel(previous_level)

    print("phase_nine_status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
