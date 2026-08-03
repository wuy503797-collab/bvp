"""Run identity, immutable metadata, and structured solver events."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

import numpy as np


RUN_METADATA_SCHEMA_VERSION = "bvp-run-v1"
APPLICATION_VERSION = "unversioned"
EVENT_DETAILS_STRING_LIMIT = 256
EVENT_LOGGER = logging.getLogger("bvp_core.events")


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp with a stable ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _plain(value: Any, *, string_limit: int | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, string_limit=string_limit)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        return [_plain(item, string_limit=string_limit) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(item, string_limit=string_limit) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item(), string_limit=string_limit)
    if isinstance(value, datetime):
        timestamp = value.astimezone(timezone.utc)
        return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Path):
        return "<redacted_path>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, str):
        return f"<{type(value).__name__} redacted>"
    text = value.replace("\r", "\\r").replace("\n", "\\n")
    sensitive_paths = {str(Path.home()), str(Path.cwd()), sys.executable}
    for sensitive in sensitive_paths:
        if sensitive:
            text = text.replace(sensitive, "<redacted_path>")
    text = re.sub(
        r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\r\n;]*",
        "<redacted_path>",
        text,
    )
    if string_limit is not None and len(text) > string_limit:
        return text[: string_limit - 3] + "..."
    return text


def sanitize_diagnostic_value(value: Any, *, string_limit: int = 512) -> Any:
    """Return bounded, path-redacted diagnostic data for logs and exports."""
    return _plain(value, string_limit=string_limit)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        kind = "nan" if math.isnan(value) else (
            "positive_infinity" if value > 0 else "negative_infinity"
        )
        return MappingProxyType({"non_finite": kind})
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"run metadata contains unsupported value {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def canonical_problem_data(problem: Any) -> dict[str, Any]:
    """Build the path- and runtime-independent problem representation."""
    return {
        "name": str(problem.name),
        "var_names": list(problem.var_names),
        "odes": list(problem.odes),
        "boundary_conditions": list(problem.boundary_conditions),
        "known_indices": list(problem.known_indices),
        "unknown_indices": list(problem.unknown_indices),
        "known_values": {
            str(index): float(problem.known_values[index])
            for index in sorted(problem.known_values)
        },
        "initial_guess": [float(value) for value in problem.initial_guess],
        "time_interval": [float(problem.t_start), float(problem.t_end)],
        "auxiliary_expressions": {
            key: problem.auxiliary_expressions[key]
            for key in sorted(problem.auxiliary_expressions)
        },
    }


def stable_problem_signature(problem: Any) -> str:
    """Return SHA-256 over canonical UTF-8 JSON, never Python ``hash()``."""
    payload = json.dumps(
        canonical_problem_data(problem),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RunContext:
    """Immutable correlation context created once for a real solve execution."""

    run_id: str
    request_id: str | None
    source_task_id: str | None
    problem_signature: str
    problem_name: str
    solver_method: str
    ivp_method: str
    started_at: str

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "problem_signature",
            "problem_name",
            "solver_method",
            "ivp_method",
            "started_at",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")

    @classmethod
    def create(
        cls,
        problem: Any,
        config: Any,
        *,
        request_id: str | None = None,
        source_task_id: str | None = None,
        run_id: str | None = None,
        started_at: str | None = None,
    ) -> "RunContext":
        return cls(
            run_id=run_id or uuid4().hex,
            request_id=request_id,
            source_task_id=source_task_id,
            problem_signature=stable_problem_signature(problem),
            problem_name=str(problem.name),
            solver_method=str(config.method),
            ivp_method=str(config.ivp_method),
            started_at=started_at or utc_now_iso(),
        )


@dataclass(frozen=True)
class RunMetadata:
    """Immutable, JSON-compatible facts describing one completed run."""

    schema_version: str
    run_id: str
    request_id: str | None
    source_task_id: str | None
    problem_signature: str
    problem_name: str
    solver_method: str
    ivp_method: str
    started_at: str
    completed_at: str
    elapsed_seconds: float
    final_status: str
    tolerance_mode: str
    effective_tolerances: Mapping[str, Any]
    python_version: str
    package_versions: Mapping[str, str]
    platform_summary: Mapping[str, str]
    application_version: str = APPLICATION_VERSION
    solver_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0 or not np.isfinite(self.elapsed_seconds):
            raise ValueError("elapsed_seconds must be a non-negative finite number")
        object.__setattr__(
            self, "effective_tolerances", _freeze(dict(self.effective_tolerances))
        )
        object.__setattr__(self, "package_versions", _freeze(dict(self.package_versions)))
        object.__setattr__(self, "platform_summary", _freeze(dict(self.platform_summary)))
        object.__setattr__(
            self, "solver_diagnostics", _freeze(dict(self.solver_diagnostics))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "source_task_id": self.source_task_id,
            "problem_signature": self.problem_signature,
            "problem_name": self.problem_name,
            "solver_method": self.solver_method,
            "ivp_method": self.ivp_method,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "final_status": self.final_status,
            "tolerance_mode": self.tolerance_mode,
            "effective_tolerances": _thaw(self.effective_tolerances),
            "python_version": self.python_version,
            "package_versions": _thaw(self.package_versions),
            "platform_summary": _thaw(self.platform_summary),
            "application_version": self.application_version,
            "solver_diagnostics": _thaw(self.solver_diagnostics),
        }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution, label in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sympy", "sympy"),
    ):
        try:
            versions[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[label] = "unavailable"
    return versions


def _solver_diagnostics(
    solver_metadata: Mapping[str, Any] | None,
    result_data: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(solver_metadata or {})
    result = dict(result_data or {})
    root = dict(metadata.get("root") or {})
    least_squares = dict(metadata.get("least_squares") or {})
    steps = list(metadata.get("steps") or [])
    return {
        "actual_path": metadata.get("optimizer", result.get("method")),
        "root": {
            "success": root.get("success"),
            "status": root.get("status"),
            "message": root.get("message"),
            "nfev": root.get("nfev"),
        },
        "least_squares": {
            "success": least_squares.get("success"),
            "status": least_squares.get("status"),
            "message": least_squares.get("message"),
            "nfev": least_squares.get("nfev"),
            "njev": least_squares.get("njev"),
            "cost": least_squares.get("cost"),
            "optimality": least_squares.get("optimality"),
        },
        "continuation": {
            "steps_requested": metadata.get("continuation_steps"),
            "steps_completed": sum(
                1 for step in steps if isinstance(step, Mapping) and step.get("newton_converged")
            ),
            "newton_updates": sum(
                int(step.get("newton_updates") or 0)
                for step in steps
                if isinstance(step, Mapping)
            ),
            "failed_step": metadata.get("failure_step"),
            "failure_reason": metadata.get("failure_reason"),
        },
        "ivp": {
            "success": result.get("ivp_success"),
            "status": result.get("ivp_status"),
            "message": result.get("ivp_message"),
            "t_final": result.get("ivp_t_final"),
        },
    }


def build_run_metadata(
    context: RunContext,
    config: Any,
    *,
    boundary_count: int,
    final_status: str,
    elapsed_seconds: float,
    solver_metadata: Mapping[str, Any] | None = None,
    result_data: Mapping[str, Any] | None = None,
    completed_at: str | None = None,
) -> RunMetadata:
    tolerances = config.effective_tolerances(boundary_count).to_metadata()
    return RunMetadata(
        schema_version=RUN_METADATA_SCHEMA_VERSION,
        run_id=context.run_id,
        request_id=context.request_id,
        source_task_id=context.source_task_id,
        problem_signature=context.problem_signature,
        problem_name=context.problem_name,
        solver_method=context.solver_method,
        ivp_method=context.ivp_method,
        started_at=context.started_at,
        completed_at=completed_at or utc_now_iso(),
        elapsed_seconds=float(elapsed_seconds),
        final_status=str(final_status),
        tolerance_mode=str(tolerances["tolerance_mode"]),
        effective_tolerances=tolerances,
        python_version=platform.python_version(),
        package_versions=_package_versions(),
        platform_summary={"system": platform.system(), "machine": platform.machine()},
        solver_diagnostics=sanitize_diagnostic_value(
            _solver_diagnostics(solver_metadata, result_data)
        ),
    )


@dataclass(frozen=True)
class SolverEvent:
    timestamp: str
    level: str
    event_name: str
    run_id: str
    request_id: str | None
    problem_signature: str
    solver_method: str
    phase: str
    status: str | None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "details",
            _freeze(_plain(dict(self.details), string_limit=EVENT_DETAILS_STRING_LIMIT)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "event_name": self.event_name,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "problem_signature": self.problem_signature,
            "solver_method": self.solver_method,
            "phase": self.phase,
            "status": self.status,
            "details": _thaw(self.details),
        }


def emit_solver_event(
    context: RunContext,
    event_name: str,
    *,
    level: int = logging.INFO,
    phase: str,
    status: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> SolverEvent:
    """Emit one structured event without configuring any global handler."""
    event = SolverEvent(
        timestamp=utc_now_iso(),
        level=logging.getLevelName(level),
        event_name=str(event_name),
        run_id=context.run_id,
        request_id=context.request_id,
        problem_signature=context.problem_signature,
        solver_method=context.solver_method,
        phase=str(phase),
        status=None if status is None else str(status),
        details=dict(details or {}),
    )
    EVENT_LOGGER.log(
        level,
        event.event_name,
        extra={
            "solver_event": event.to_dict(),
            "run_id": event.run_id,
            "request_id": event.request_id,
            "event_name": event.event_name,
        },
    )
    return event


def metadata_context(metadata: RunMetadata) -> RunContext:
    """Reconstruct correlation-only context for post-solve GUI/export events."""
    return RunContext(
        run_id=metadata.run_id,
        request_id=metadata.request_id,
        source_task_id=metadata.source_task_id,
        problem_signature=metadata.problem_signature,
        problem_name=metadata.problem_name,
        solver_method=metadata.solver_method,
        ivp_method=metadata.ivp_method,
        started_at=metadata.started_at,
    )


def attach_run_metadata(exc: BaseException, metadata: RunMetadata) -> None:
    """Best-effort association for exceptions crossing the public API boundary."""
    try:
        setattr(exc, "run_metadata", metadata)
    except Exception:
        pass
