"""GUI-independent solve request, provenance, and lifecycle models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Event
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from .models import BVPProblem, SolverConfig
from .observability import (
    RunMetadata,
    canonical_problem_data,
    stable_problem_signature,
)
from .results import BVPResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _readonly_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1).copy()
    array.setflags(write=False)
    return array


def _readonly_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if array.ndim != 2:
        raise ValueError("plot state data must have shape (state, sample)")
    array.setflags(write=False)
    return array


def _freeze_primary_plot_snapshot(result: BVPResult) -> tuple[np.ndarray, np.ndarray]:
    """Sample dense output once, then detach plotting from the SciPy object."""
    sample_t = np.asarray(result.t, dtype=float)
    sample_y = np.asarray(result.y, dtype=float)
    dense_solution = result.sol
    dense_function = getattr(dense_solution, "sol", None)
    if sample_t.size >= 2 and callable(dense_function):
        candidate_t = np.linspace(float(sample_t[0]), float(sample_t[-1]), 500)
        try:
            candidate_y = np.asarray(dense_function(candidate_t), dtype=float)
        except Exception:
            candidate_y = np.empty((0, 0), dtype=float)
        if (
            candidate_y.shape == (result.y.shape[0], candidate_t.size)
            and np.isfinite(candidate_t).all()
            and np.isfinite(candidate_y).all()
        ):
            sample_t = candidate_t
            sample_y = candidate_y
    return _readonly_vector(sample_t), _readonly_matrix(sample_y)


def _freeze_plain_metadata(value: Any, *, path: str = "display_metadata") -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_plain_metadata(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_plain_metadata(item, path=path) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"{path} contains unsupported value {type(value).__name__}; "
        "Qt and other opaque objects are not allowed"
    )


class SolveOutcomeStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GuiSolveState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_GUI_TRANSITIONS = {
    GuiSolveState.IDLE: {GuiSolveState.RUNNING},
    GuiSolveState.RUNNING: {
        GuiSolveState.COMPLETED,
        GuiSolveState.FAILED,
        GuiSolveState.CANCEL_REQUESTED,
        GuiSolveState.CANCELLED,
    },
    GuiSolveState.CANCEL_REQUESTED: {
        GuiSolveState.COMPLETED,
        GuiSolveState.FAILED,
        GuiSolveState.CANCELLED,
    },
    GuiSolveState.COMPLETED: {GuiSolveState.IDLE, GuiSolveState.RUNNING},
    GuiSolveState.FAILED: {GuiSolveState.IDLE, GuiSolveState.RUNNING},
    GuiSolveState.CANCELLED: {GuiSolveState.IDLE, GuiSolveState.RUNNING},
}


def gui_transition_allowed(current: GuiSolveState, target: GuiSolveState) -> bool:
    """Return whether one explicit GUI lifecycle transition is valid."""
    return current == target or target in _GUI_TRANSITIONS[current]


class SolveCancelled(RuntimeError):
    """Raised at a cooperative checkpoint, optionally with completed run facts."""

    def __init__(
        self, message: str, *, run_metadata: RunMetadata | None = None
    ) -> None:
        self.run_metadata = run_metadata
        super().__init__(message)


class CancellationToken:
    """Thread-safe cooperative cancellation token based on ``threading.Event``."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise SolveCancelled("The solve request was cancelled cooperatively.")

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation; useful for controlled worker lifecycle tests."""
        return self._event.wait(timeout)


@dataclass(frozen=True)
class SolveRequest:
    """Immutable snapshot of one GUI solve request and its display provenance."""

    request_id: str
    problem: BVPProblem
    config: SolverConfig
    source_task_id: str
    source_task_index: int | None = None
    display_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.source_task_id:
            raise ValueError("source_task_id must be non-empty")
        if not isinstance(self.problem, BVPProblem):
            raise TypeError("problem must be a BVPProblem")
        if not isinstance(self.config, SolverConfig):
            raise TypeError("config must be a SolverConfig")
        if self.source_task_index is not None and self.source_task_index < 0:
            raise ValueError("source_task_index must be non-negative when present")
        object.__setattr__(
            self,
            "display_metadata",
            _freeze_plain_metadata(dict(self.display_metadata)),
        )

    @classmethod
    def create(
        cls,
        *,
        problem: BVPProblem,
        config: SolverConfig,
        source_task_id: str,
        source_task_index: int | None = None,
        display_metadata: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        created_at: datetime | None = None,
    ) -> "SolveRequest":
        return cls(
            request_id=request_id or uuid4().hex,
            problem=problem,
            config=config,
            source_task_id=source_task_id,
            source_task_index=source_task_index,
            display_metadata=dict(display_metadata or {}),
            created_at=created_at or _utc_now(),
        )

    @property
    def task_name(self) -> str:
        return self.problem.name

    @property
    def var_names(self) -> tuple[str, ...]:
        return self.problem.var_names

    @property
    def known_indices(self) -> tuple[int, ...]:
        return self.problem.known_indices

    @property
    def unknown_indices(self) -> tuple[int, ...]:
        return self.problem.unknown_indices

    @property
    def auxiliary_expressions(self) -> Mapping[str, str]:
        return self.problem.auxiliary_expressions

    @property
    def state_dimension(self) -> int:
        return self.problem.state_dimension

    @property
    def time_interval(self) -> tuple[float, float]:
        return self.problem.t_start, self.problem.t_end

    @property
    def problem_signature(self) -> str:
        """Stable cross-process SHA-256 used for provenance and plot isolation."""
        return stable_problem_signature(self.problem)

    def build_initial_state(self, p_opt: np.ndarray) -> np.ndarray:
        """Build the full initial state exclusively from this request snapshot."""
        parameters = np.asarray(p_opt, dtype=float).reshape(-1)
        state = np.zeros(self.state_dimension, dtype=float)
        for index in self.known_indices:
            state[index] = self.problem.known_values[index]
        for parameter_index, state_index in enumerate(self.unknown_indices):
            if parameter_index < parameters.size:
                state[state_index] = parameters[parameter_index]
        return state


@dataclass(frozen=True)
class SolveOutcome:
    """One worker terminal event, including failures that have no BVP result."""

    request: SolveRequest
    status: SolveOutcomeStatus
    result: BVPResult | None = None
    message: str = ""
    technical_diagnostic: str = ""
    run_metadata: RunMetadata | None = None
    completed_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SolveOutcomeStatus(self.status))
        metadata = self.run_metadata
        if metadata is None and self.result is not None:
            metadata = self.result.run_metadata
            object.__setattr__(self, "run_metadata", metadata)
        if metadata is not None:
            if not isinstance(metadata, RunMetadata):
                raise TypeError("run_metadata must be a RunMetadata instance or None")
            if metadata.request_id not in {None, self.request.request_id}:
                raise ValueError("run_metadata request_id does not match outcome request")
            if metadata.problem_signature != self.request.problem_signature:
                raise ValueError(
                    "run_metadata problem_signature does not match outcome request"
                )
        if self.status is SolveOutcomeStatus.COMPLETED:
            if self.result is None or not self.result.success:
                raise ValueError("completed outcome requires a successful BVPResult")
        elif self.status is SolveOutcomeStatus.FAILED:
            if self.result is not None and self.result.success:
                raise ValueError("failed outcome cannot contain a successful BVPResult")
        elif self.result is not None:
            raise ValueError("cancelled outcome cannot contain a numerical result")

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def run_id(self) -> str | None:
        return None if self.run_metadata is None else self.run_metadata.run_id


@dataclass(frozen=True)
class SolveRecord:
    """Successful result permanently bound to its immutable request snapshot."""

    request: SolveRequest
    result: BVPResult
    auxiliary_outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    auxiliary_sample_t: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    auxiliary_errors: tuple[str, ...] | list[str] = field(default_factory=tuple)
    completed_at: datetime = field(default_factory=_utc_now)
    primary_plot_t: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    primary_plot_y: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )

    def __post_init__(self) -> None:
        if not self.result.success:
            raise ValueError("SolveRecord only stores successful results")
        if self.result.y.shape[0] != self.request.state_dimension:
            raise ValueError(
                "result state dimension does not match the request snapshot"
            )
        if self.result.y.shape[1] != self.result.t.size:
            raise ValueError("result y sample count does not match result t")
        if (
            self.result.run_metadata is not None
            and self.result.run_metadata.problem_signature
            != self.request.problem_signature
        ):
            raise ValueError(
                "result run_metadata problem_signature does not match request"
            )
        sample_t = _readonly_vector(self.auxiliary_sample_t)
        frozen_outputs: dict[str, np.ndarray] = {}
        for name, values in self.auxiliary_outputs.items():
            if not isinstance(name, str) or not name:
                raise ValueError("auxiliary output names must be non-empty strings")
            array = _readonly_vector(values)
            if sample_t.size and array.size != sample_t.size:
                raise ValueError(
                    f"auxiliary output {name!r} does not match its sample grid"
                )
            frozen_outputs[name] = array
        if frozen_outputs and not sample_t.size:
            raise ValueError("auxiliary outputs require an explicit sample grid")
        plot_t = np.asarray(self.primary_plot_t, dtype=float).reshape(-1)
        plot_y = np.asarray(self.primary_plot_y, dtype=float)
        if not plot_t.size and not plot_y.size:
            plot_t, plot_y = _freeze_primary_plot_snapshot(self.result)
        else:
            plot_t = _readonly_vector(plot_t)
            plot_y = _readonly_matrix(plot_y)
        if plot_y.shape != (self.request.state_dimension, plot_t.size):
            raise ValueError(
                "primary plot snapshot must match the request state dimension "
                "and plot sample count"
            )
        if not np.isfinite(plot_t).all() or not np.isfinite(plot_y).all():
            raise ValueError("primary plot snapshot must contain only finite values")
        object.__setattr__(self, "auxiliary_sample_t", sample_t)
        object.__setattr__(self, "auxiliary_outputs", MappingProxyType(frozen_outputs))
        object.__setattr__(self, "auxiliary_errors", tuple(self.auxiliary_errors))
        object.__setattr__(self, "primary_plot_t", plot_t)
        object.__setattr__(self, "primary_plot_y", plot_y)

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def run_id(self) -> str | None:
        metadata = self.result.run_metadata
        return None if metadata is None else metadata.run_id

    def to_legacy_dict(self) -> dict[str, Any]:
        """Build a defensive dictionary only at a legacy GUI/export boundary."""
        data = self.result.to_dict()
        data.update(
            {
                "request_id": self.request_id,
                "source_task_id": self.request.source_task_id,
                "task_name": self.request.task_name,
                "var_names": list(self.request.var_names),
                "aux": {
                    name: values.copy()
                    for name, values in self.auxiliary_outputs.items()
                },
                "aux_names": list(self.auxiliary_outputs),
                "aux_t": self.auxiliary_sample_t.copy(),
                "auxiliary_errors": list(self.auxiliary_errors),
            }
        )
        return data


@dataclass(frozen=True)
class PlotCompatibility:
    compatible: bool
    reason: str


def plot_compatibility(
    anchor: SolveRecord, candidate: SolveRecord
) -> PlotCompatibility:
    """Check whether two records may be intentionally overlaid.

    Results from one stable GUI task may be compared after any numerical input
    or solver-control edit. Separate tasks remain isolated unless their problem
    snapshots differ only by the initial guess. In every case the plotted state
    coordinates must retain the same names and dimension.
    """
    for record in (anchor, candidate):
        if record.result.y.ndim != 2:
            return PlotCompatibility(False, "result state data is not two-dimensional")
        if record.result.y.shape[0] != record.request.state_dimension:
            return PlotCompatibility(False, "result state dimension is invalid")
        if record.result.y.shape[1] != record.result.t.size:
            return PlotCompatibility(False, "result time and state samples differ")
    if anchor.request.state_dimension != candidate.request.state_dimension:
        return PlotCompatibility(False, "state dimensions differ")
    if anchor.request.var_names != candidate.request.var_names:
        return PlotCompatibility(False, "variable names or meanings differ")
    if anchor.request.source_task_id == candidate.request.source_task_id:
        return PlotCompatibility(True, "compatible variants of one source task")
    anchor_problem = canonical_problem_data(anchor.request.problem)
    candidate_problem = canonical_problem_data(candidate.request.problem)
    anchor_problem.pop("initial_guess", None)
    candidate_problem.pop("initial_guess", None)
    if anchor_problem != candidate_problem:
        return PlotCompatibility(False, "source tasks or problem definitions differ")
    return PlotCompatibility(True, "compatible problem snapshot")


def partition_plot_records(
    records: Sequence[SolveRecord], anchor: SolveRecord
) -> tuple[tuple[SolveRecord, ...], tuple[tuple[SolveRecord, str], ...]]:
    compatible: list[SolveRecord] = []
    rejected: list[tuple[SolveRecord, str]] = []
    for record in records:
        check = plot_compatibility(anchor, record)
        if check.compatible:
            compatible.append(record)
        else:
            rejected.append((record, check.reason))
    return tuple(compatible), tuple(rejected)
