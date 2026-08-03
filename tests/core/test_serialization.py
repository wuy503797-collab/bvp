"""Strict canonical serialization and atomic write contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from bvp_core import (
    BVPProblem,
    SolveOutcome,
    SolveOutcomeStatus,
    SolveRecord,
    SolveRequest,
    SolverConfig,
    build_canonical_export_record,
    canonical_json_text,
    canonical_text_summary,
    normalize_json_value,
    solve_bvp_problem,
    write_canonical_json,
    write_canonical_text,
)


def _request(name: str = "中文 / Русский") -> SolveRequest:
    problem = BVPProblem(
        name=name,
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
        auxiliary_expressions={"двойной": "2*x"},
    )
    return SolveRequest.create(
        problem=problem,
        config=SolverConfig(method="shooting"),
        source_task_id="task-unicode",
    )


def _record() -> SolveRecord:
    request = _request()
    result = solve_bvp_problem(
        request.problem,
        request.config,
        run_context=__import__("bvp_core").RunContext.create(
            request.problem,
            request.config,
            request_id=request.request_id,
            source_task_id=request.source_task_id,
        ),
    )
    return SolveRecord(
        request=request,
        result=result,
        auxiliary_outputs={"двойной": 2.0 * result.y[0]},
        auxiliary_sample_t=result.t,
        auxiliary_errors=["辅助 ошибка: simplified"],
        completed_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
    )


def test_normalizer_handles_supported_types_and_marks_non_finite_values() -> None:
    normalized = normalize_json_value(
        {
            "array": np.array([1.0, np.nan, np.inf, -np.inf]),
            "scalar": np.int64(3),
            "mapping": MappingProxyType({"ok": np.bool_(True)}),
            "tuple": (1, 2),
            "time": datetime(2026, 8, 3, tzinfo=timezone.utc),
            "path": Path("relative.txt"),
            "none": None,
        }
    )

    assert normalized["array"] == [
        1.0,
        {"non_finite": "nan"},
        {"non_finite": "positive_infinity"},
        {"non_finite": "negative_infinity"},
    ]
    assert normalized["scalar"] == 3
    assert normalized["mapping"] == {"ok": True}
    assert normalized["time"].endswith("Z")


def test_normalizer_rejects_opaque_scipy_or_qt_style_objects() -> None:
    with pytest.raises(TypeError, match="opaque"):
        normalize_json_value(object())


def test_result_normalization_excludes_raw_scipy_solution_object() -> None:
    result = _record().result

    normalized = normalize_json_value(result)

    assert "raw_solution" not in normalized
    assert normalized["success"] is True
    assert normalized["run_metadata"]["run_id"] == result.run_metadata.run_id


def test_success_record_json_is_strict_complete_unicode_and_deterministic() -> None:
    record = build_canonical_export_record(_record())

    first = canonical_json_text(record)
    second = canonical_json_text(record)
    decoded = json.loads(first)

    assert first == second
    assert decoded["schema_version"] == "bvp-result-v1"
    assert decoded["problem"]["name"] == "中文 / Русский"
    assert decoded["run"]["run_id"]
    assert decoded["problem"]["problem_signature"] == decoded["run"][
        "problem_signature"
    ]
    assert decoded["result"]["success"] is True
    assert decoded["diagnostics"]["boundary_acceptance"]["thresholds"]
    assert decoded["auxiliary"]["outputs"]["двойной"]
    assert "NaN" not in first and "Infinity" not in first


def test_text_and_json_share_key_facts() -> None:
    record = build_canonical_export_record(_record())
    data = json.loads(canonical_json_text(record))
    text = canonical_text_summary(record)

    for value in (
        data["schema_version"],
        data["run"]["run_id"],
        data["problem"]["problem_signature"],
        data["problem"]["name"],
        data["configuration"]["method"],
        data["result"]["status"],
        str(data["run"]["elapsed_seconds"]),
    ):
        assert value in text
    assert "boundary_thresholds:" in text
    assert "effective_tolerances:" in text


def test_failure_and_cancelled_outcomes_have_distinct_record_types() -> None:
    request = _request("Failure")
    failed_result = solve_bvp_problem(
        BVPProblem(
            name="No root",
            odes=["0"],
            var_names=["x"],
            boundary_conditions=["x0_T**2 + 1"],
            known_indices=[],
            unknown_indices=[0],
            known_values={},
            initial_guess=[0.0],
        ),
        SolverConfig(method="shooting"),
    )
    failed = SolveOutcome(
        request=SolveRequest.create(
            problem=BVPProblem(
                name="No root",
                odes=["0"],
                var_names=["x"],
                boundary_conditions=["x0_T**2 + 1"],
                known_indices=[],
                unknown_indices=[0],
                known_values={},
                initial_guess=[0.0],
            ),
            config=SolverConfig(method="shooting"),
            source_task_id="failure",
        ),
        status=SolveOutcomeStatus.FAILED,
        result=failed_result,
        message=failed_result.message,
    )
    cancelled = SolveOutcome(
        request=request,
        status=SolveOutcomeStatus.CANCELLED,
        message="cancelled",
    )

    failed_data = build_canonical_export_record(failed).to_dict()
    cancelled_data = build_canonical_export_record(cancelled).to_dict()
    assert failed_data["record_type"] == "failure_diagnostic"
    assert failed_data["result"]["success"] is False
    assert cancelled_data["record_type"] == "cancelled"
    assert cancelled_data["result"]["status"] == "cancelled"


def test_atomic_json_and_text_write_support_unicode_paths_and_replacement(tmp_path) -> None:
    record = build_canonical_export_record(_record())
    target_dir = tmp_path / "结果 каталог"
    json_path = target_dir / "结果.json"
    text_path = target_dir / "摘要.txt"
    json_path.parent.mkdir(parents=True)
    json_path.write_text("old", encoding="utf-8")

    write_canonical_json(json_path, record)
    write_canonical_text(text_path, record)

    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == (
        "bvp-result-v1"
    )
    assert text_path.read_text(encoding="utf-8").startswith(
        "schema_version: bvp-result-v1"
    )
    assert not list(target_dir.glob(".*.tmp"))


def test_serialization_failure_preserves_existing_target(tmp_path) -> None:
    from bvp_core.serialization import write_json_data_atomic

    target = tmp_path / "existing.json"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_data_atomic(target, {"opaque": object()})

    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".*.tmp"))


def test_mid_write_failure_preserves_existing_target_and_cleans_temp(
    tmp_path, monkeypatch
) -> None:
    from bvp_core import serialization

    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    def fail_after_prefix(handle, _text):
        handle.write("partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(serialization, "_write_text_payload", fail_after_prefix)
    with pytest.raises(OSError, match="simulated"):
        serialization.atomic_write_text(target, "replacement")

    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".*.tmp"))


def test_replace_failure_preserves_existing_target_and_cleans_temp(
    tmp_path, monkeypatch
) -> None:
    from bvp_core import serialization

    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        serialization.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        serialization.atomic_write_text(target, "replacement")

    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".*.tmp"))


def test_failure_export_redacts_machine_paths_but_problem_snapshot_is_complete() -> None:
    request = _request("Failure privacy")
    outcome = SolveOutcome(
        request=request,
        status=SolveOutcomeStatus.FAILED,
        message=f"internal failure at {Path.home()}",
    )

    text = canonical_json_text(build_canonical_export_record(outcome))
    data = json.loads(text)

    assert str(Path.home()) not in text
    assert data["result"]["message"] == "internal failure at <redacted_path>"
    assert data["problem"]["odes"] == ["x"]
