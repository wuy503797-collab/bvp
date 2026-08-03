"""Canonical result snapshots, strict JSON, text summaries, and atomic writes."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .observability import canonical_problem_data, sanitize_diagnostic_value
from .requests import SolveOutcome, SolveOutcomeStatus, SolveRecord, SolveRequest
from .results import BVPResult
from .tolerances import PROCESS_TOLERANCE_FIELDS


EXPORT_SCHEMA_VERSION = "bvp-result-v1"
_AUXILIARY_ERROR_LIMIT = 512


def _non_finite(value: float) -> dict[str, str]:
    if math.isnan(value):
        kind = "nan"
    elif value > 0:
        kind = "positive_infinity"
    else:
        kind = "negative_infinity"
    return {"non_finite": kind}


def normalize_json_value(value: Any) -> Any:
    """Normalize supported values without serializing opaque runtime objects."""
    if isinstance(value, np.ndarray):
        return normalize_json_value(value.tolist())
    if isinstance(value, np.generic):
        return normalize_json_value(value.item())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(
                    "canonical JSON mapping keys must be strings or integers"
                )
            normalized[str(key)] = normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
    if isinstance(value, Enum):
        return normalize_json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else _non_finite(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if is_dataclass(value):
        return {
            item.name: normalize_json_value(getattr(value, item.name))
            for item in fields(value)
            if item.repr
        }
    raise TypeError(
        f"unsupported canonical serialization value {type(value).__name__}; "
        "opaque SciPy, Qt, function, and SymPy objects are not serializable"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class CanonicalExportRecord:
    """One immutable export fact tree shared by JSON and text renderers."""

    schema_version: str
    record_type: str
    problem: Mapping[str, Any]
    configuration: Mapping[str, Any]
    effective_tolerances: Mapping[str, Any]
    run: Mapping[str, Any]
    result: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    auxiliary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != EXPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported export schema {self.schema_version!r}")
        for name in (
            "problem",
            "configuration",
            "effective_tolerances",
            "run",
            "result",
            "diagnostics",
            "provenance",
            "auxiliary",
        ):
            normalized = normalize_json_value(dict(getattr(self, name)))
            object.__setattr__(self, name, _freeze(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "problem": _thaw(self.problem),
            "configuration": _thaw(self.configuration),
            "effective_tolerances": _thaw(self.effective_tolerances),
            "run": _thaw(self.run),
            "result": _thaw(self.result),
            "diagnostics": _thaw(self.diagnostics),
            "provenance": _thaw(self.provenance),
            "auxiliary": _thaw(self.auxiliary),
        }


def _configuration_snapshot(request: SolveRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    config = request.config
    effective = config.effective_tolerances(
        len(request.problem.boundary_conditions)
    ).to_metadata()
    explicit = {
        name: config.explicit_tolerance_value(name)
        for name in PROCESS_TOLERANCE_FIELDS
    }
    configuration = {
        "method": config.method,
        "ivp_method": config.ivp_method,
        "legacy_eps": config.eps,
        "explicit_tolerances": explicit,
        "tolerance_mode": config.tolerance_mode,
        "continuation_steps": config.continuation_steps,
        "boundary_atol": config.boundary_atol,
        "boundary_rtol": config.boundary_rtol,
        "boundary_scales": (
            None if config.boundary_scales is None else list(config.boundary_scales)
        ),
    }
    return configuration, effective


def _problem_snapshot(request: SolveRequest) -> dict[str, Any]:
    snapshot = canonical_problem_data(request.problem)
    snapshot["problem_signature"] = request.problem_signature
    return snapshot


def _result_snapshot(result: BVPResult | None, status: str, message: str) -> dict[str, Any]:
    if result is None:
        return {
            "success": False,
            "status": status,
            "message": sanitize_diagnostic_value(message),
            "method": None,
            "p_opt": [],
            "t": [],
            "y": [],
            "boundary_residual": [],
            "boundary_residual_norm": None,
            "iterations": None,
        }
    return {
        "success": result.success,
        "status": result.status,
        "message": sanitize_diagnostic_value(result.message),
        "method": result.method,
        "p_opt": result.p_opt,
        "t": result.t,
        "y": result.y,
        "boundary_residual": result.boundary_residual,
        "boundary_residual_norm": result.boundary_residual_norm,
        "iterations": result.iterations,
    }


def _diagnostics_snapshot(result: BVPResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "ivp": None,
            "optimizer": None,
            "algorithm": None,
            "boundary_acceptance": None,
            "solver_metadata": {},
        }
    diagnostics = {
        "ivp": {
            "success": result.ivp_success,
            "status": result.ivp_status,
            "message": result.ivp_message,
            "t_final": result.ivp_t_final,
        },
        "optimizer": {"success": result.optimizer_success},
        "algorithm": {
            "success": result.algorithm_success,
            "finite_success": result.finite_success,
        },
        "boundary_acceptance": {
            "success": result.boundary_success,
            "formula": result.boundary_acceptance,
            "atol": result.boundary_atol,
            "rtol": result.boundary_rtol,
            "scales": result.boundary_scales,
            "thresholds": result.boundary_thresholds,
            "component_success": result.boundary_component_success,
            "scaled_ratios": result.boundary_scaled_ratios,
            "max_scaled_ratio": result.boundary_max_scaled_ratio,
        },
        "solver_metadata": result.solver_metadata,
    }
    return sanitize_diagnostic_value(diagnostics)


def _auxiliary_error(error: Any) -> dict[str, Any]:
    if isinstance(error, Mapping):
        name = error.get("name")
        code = error.get("error_code", "auxiliary_output_failed")
        reason = error.get("reason", error.get("message", ""))
    else:
        name = None
        code = "auxiliary_output_failed"
        reason = str(error)
    reason = sanitize_diagnostic_value(
        str(reason), string_limit=_AUXILIARY_ERROR_LIMIT
    )
    return {"name": name, "error_code": str(code), "reason": reason}


def build_canonical_export_record(
    source: SolveRecord | SolveOutcome,
) -> CanonicalExportRecord:
    """Build the sole export fact tree from an immutable request snapshot."""
    if isinstance(source, SolveRecord):
        request = source.request
        result = source.result
        status = result.status
        message = result.message
        record_type = "result"
        metadata = result.run_metadata
        auxiliary = {
            "sample_t": source.auxiliary_sample_t,
            "outputs": source.auxiliary_outputs,
            "errors": [_auxiliary_error(error) for error in source.auxiliary_errors],
            "sample_count": int(source.auxiliary_sample_t.size),
        }
        completed_at = source.completed_at
    elif isinstance(source, SolveOutcome):
        request = source.request
        result = source.result
        status = (
            "cancelled"
            if source.status is SolveOutcomeStatus.CANCELLED
            else (result.status if result is not None else "internal_error")
        )
        message = source.message
        record_type = (
            "cancelled"
            if source.status is SolveOutcomeStatus.CANCELLED
            else "failure_diagnostic"
        )
        metadata = source.run_metadata
        auxiliary = {"sample_t": [], "outputs": {}, "errors": [], "sample_count": 0}
        completed_at = source.completed_at
    else:
        raise TypeError("source must be a SolveRecord or SolveOutcome")

    configuration, effective = _configuration_snapshot(request)
    run = (
        metadata.to_dict()
        if metadata is not None
        else {
            "schema_version": None,
            "run_id": None,
            "request_id": request.request_id,
            "source_task_id": request.source_task_id,
            "problem_signature": request.problem_signature,
            "problem_name": request.task_name,
            "solver_method": request.config.method,
            "ivp_method": request.config.ivp_method,
            "started_at": None,
            "completed_at": normalize_json_value(completed_at),
            "elapsed_seconds": None,
            "final_status": status,
        }
    )
    return CanonicalExportRecord(
        schema_version=EXPORT_SCHEMA_VERSION,
        record_type=record_type,
        problem=_problem_snapshot(request),
        configuration=configuration,
        effective_tolerances=effective,
        run=run,
        result=_result_snapshot(result, status, message),
        diagnostics=_diagnostics_snapshot(result),
        provenance={
            "request_id": request.request_id,
            "source_task_id": request.source_task_id,
            "source_task_index": request.source_task_index,
            "request_created_at": request.created_at,
            "display_metadata": request.display_metadata,
        },
        auxiliary=auxiliary,
    )


def canonical_json_text(record: CanonicalExportRecord) -> str:
    """Render strict, deterministic, UTF-8-ready JSON."""
    return (
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )


def _text_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def canonical_text_summary(record: CanonicalExportRecord) -> str:
    """Render a human-readable summary from the same canonical fact tree."""
    data = record.to_dict()
    problem = data["problem"]
    config = data["configuration"]
    run = data["run"]
    result = data["result"]
    boundary = data["diagnostics"].get("boundary_acceptance") or {}
    ivp = data["diagnostics"].get("ivp")
    optimizer = data["diagnostics"].get("optimizer")
    algorithm = data["diagnostics"].get("algorithm")
    lines = [
        f"schema_version: {data['schema_version']}",
        f"record_type: {data['record_type']}",
        f"run_id: {_text_value(run.get('run_id'))}",
        f"request_id: {_text_value(run.get('request_id'))}",
        f"problem_signature: {problem.get('problem_signature')}",
        "",
        "[problem]",
        f"name: {problem.get('name')}",
        f"var_names: {_text_value(problem.get('var_names'))}",
        f"odes: {_text_value(problem.get('odes'))}",
        f"boundary_conditions: {_text_value(problem.get('boundary_conditions'))}",
        f"time_interval: {_text_value(problem.get('time_interval'))}",
        "",
        "[configuration]",
        f"method: {config.get('method')}",
        f"ivp_method: {config.get('ivp_method')}",
        f"tolerance_mode: {config.get('tolerance_mode')}",
        f"legacy_eps: {_text_value(config.get('legacy_eps'))}",
        f"continuation_steps: {_text_value(config.get('continuation_steps'))}",
        f"boundary_scales: {_text_value(config.get('boundary_scales'))}",
        f"explicit_tolerances: {_text_value(config.get('explicit_tolerances'))}",
        f"effective_tolerances: {_text_value(data['effective_tolerances'])}",
        "",
        "[run]",
        f"final_status: {run.get('final_status')}",
        f"started_at: {_text_value(run.get('started_at'))}",
        f"completed_at: {_text_value(run.get('completed_at'))}",
        f"elapsed_seconds: {_text_value(run.get('elapsed_seconds'))}",
        "",
        "[result]",
        f"success: {_text_value(result.get('success'))}",
        f"status: {result.get('status')}",
        f"message: {result.get('message')}",
        f"boundary_residual: {_text_value(result.get('boundary_residual'))}",
        f"boundary_residual_norm: {_text_value(result.get('boundary_residual_norm'))}",
        f"boundary_thresholds: {_text_value(boundary.get('thresholds'))}",
        f"boundary_max_scaled_ratio: {_text_value(boundary.get('max_scaled_ratio'))}",
        "",
        "[diagnostics]",
        f"ivp: {_text_value(ivp)}",
        f"optimizer: {_text_value(optimizer)}",
        f"algorithm: {_text_value(algorithm)}",
        f"boundary_acceptance: {_text_value(boundary)}",
        f"solver_metadata: {_text_value(data['diagnostics'].get('solver_metadata'))}",
        "",
        "[data]",
    ]
    t_values = result.get("t") or []
    y_values = result.get("y") or []
    var_names = problem.get("var_names") or []
    lines.append("t" + "".join(f"\t{name}" for name in var_names))
    for index, time_value in enumerate(t_values):
        row = [_text_value(time_value)]
        for state_values in y_values:
            row.append(_text_value(state_values[index]))
        lines.append("\t".join(row))
    lines.extend(
        [
            "",
            "[auxiliary]",
            f"sample_t: {_text_value(data['auxiliary'].get('sample_t'))}",
            f"outputs: {_text_value(data['auxiliary'].get('outputs'))}",
            f"errors: {_text_value(data['auxiliary'].get('errors'))}",
            "",
            "# Human-readable snapshot; JSON is the machine-readable fact source.",
        ]
    )
    return "\n".join(lines)


def _write_text_payload(handle: Any, text: str) -> None:
    handle.write(text)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace ``path`` using a same-directory UTF-8 temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            _write_text_payload(handle, text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def write_json_data_atomic(path: str | Path, data: Any) -> None:
    """Strict atomic JSON writer also used by the task-library save path."""
    normalized = normalize_json_value(data)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    atomic_write_text(path, text)


def write_canonical_json(path: str | Path, record: CanonicalExportRecord) -> None:
    atomic_write_text(path, canonical_json_text(record))


def write_canonical_text(path: str | Path, record: CanonicalExportRecord) -> None:
    atomic_write_text(path, canonical_text_summary(record))


def export_canonical_record(
    path: str | Path, record: CanonicalExportRecord
) -> None:
    target = Path(path)
    if target.suffix.lower() == ".json":
        write_canonical_json(target, record)
    else:
        write_canonical_text(target, record)
