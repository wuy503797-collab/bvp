"""Stable, GUI-independent result model for the public BVP API."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .observability import RunMetadata


def _readonly_vector(value: Any) -> np.ndarray:
    array = np.array([] if value is None else value, dtype=float, copy=True).reshape(-1)
    array.setflags(write=False)
    return array


def _readonly_bool_vector(value: Any) -> np.ndarray:
    array = np.array([] if value is None else value, dtype=bool, copy=True).reshape(-1)
    array.setflags(write=False)
    return array


def _readonly_state_matrix(value: Any) -> np.ndarray:
    array = np.array([] if value is None else value, dtype=float, copy=True)
    if array.size == 0:
        array = np.empty((0, 0), dtype=float)
    elif array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError("result y must have shape (state, sample)")
    array.setflags(write=False)
    return array


def _freeze_metadata(value: Any, *, path: str = "solver_metadata") -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_metadata(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_metadata(item, path=path) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item, path=path) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"{path} contains unsupported value {type(value).__name__}; "
        "metadata must contain plain diagnostic data"
    )


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


@dataclass(frozen=True)
class BVPResult:
    """Structured public result for both successful and failed BVP solves.

    Array fields are copied and marked read-only. ``to_dict`` returns fresh array
    copies for legacy consumers. ``raw_solution`` is the optional SciPy compatibility
    object; public status and numerical arrays do not depend on it.
    """

    success: bool
    status: str
    message: str
    method: str
    p_opt: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    t: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    y: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    ivp_success: bool = False
    ivp_status: int | None = None
    ivp_message: str = ""
    ivp_t_final: float | None = None
    optimizer_success: bool = False
    algorithm_success: bool = False
    finite_success: bool = False
    boundary_success: bool = False
    boundary_residual: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    boundary_residual_norm: float = float("inf")
    boundary_atol: float = 1e-8
    boundary_rtol: float = 0.0
    boundary_scales: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    boundary_thresholds: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    boundary_component_success: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    boundary_scaled_ratios: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    boundary_max_scaled_ratio: float | None = None
    iterations: int = 0
    solver_metadata: Mapping[str, Any] = field(default_factory=dict)
    run_metadata: RunMetadata | None = None
    boundary_acceptance: str = (
        "abs(residual_i) <= boundary_atol + boundary_rtol * boundary_scales_i"
    )
    raw_solution: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("result status must be a non-empty machine-readable string")
        if not isinstance(self.message, str):
            raise TypeError("result message must be a string")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("result method must be a non-empty string")
        object.__setattr__(self, "p_opt", _readonly_vector(self.p_opt))
        object.__setattr__(self, "t", _readonly_vector(self.t))
        object.__setattr__(self, "y", _readonly_state_matrix(self.y))
        object.__setattr__(
            self, "boundary_residual", _readonly_vector(self.boundary_residual)
        )
        residual = self.boundary_residual
        scales = _readonly_vector(self.boundary_scales)
        if residual.size and not scales.size:
            scales = _readonly_vector(np.ones(residual.size, dtype=float))
        thresholds = _readonly_vector(self.boundary_thresholds)
        if residual.size and not thresholds.size:
            thresholds = _readonly_vector(
                float(self.boundary_atol) + float(self.boundary_rtol) * scales
            )
        component_success = _readonly_bool_vector(self.boundary_component_success)
        if residual.size and not component_success.size:
            component_success = _readonly_bool_vector(
                np.isfinite(residual) & (np.abs(residual) <= thresholds)
            )
        ratios = _readonly_vector(self.boundary_scaled_ratios)
        if residual.size and not ratios.size:
            ratios = _readonly_vector(
                np.where(np.isfinite(residual), np.abs(residual) / thresholds, np.inf)
            )
        for name, array in (
            ("boundary_scales", scales),
            ("boundary_thresholds", thresholds),
            ("boundary_component_success", component_success),
            ("boundary_scaled_ratios", ratios),
        ):
            if residual.size and array.size != residual.size:
                raise ValueError(f"{name} must match boundary_residual length")
            object.__setattr__(self, name, array)
        maximum = self.boundary_max_scaled_ratio
        if maximum is None:
            maximum = float(np.max(ratios)) if ratios.size else float("inf")
        object.__setattr__(self, "boundary_max_scaled_ratio", float(maximum))
        object.__setattr__(
            self, "solver_metadata", _freeze_metadata(self.solver_metadata)
        )
        if self.run_metadata is not None and not isinstance(
            self.run_metadata, RunMetadata
        ):
            raise TypeError("run_metadata must be a RunMetadata instance or None")

    @property
    def sol(self) -> Any | None:
        """Optional SciPy compatibility object used for dense output."""
        return self.raw_solution

    @property
    def residual_norm(self) -> float:
        """Backward-compatible alias for the final boundary residual norm."""
        return self.boundary_residual_norm

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy result mapping with defensive array copies."""
        return {
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "method": self.method,
            "p_opt": self.p_opt.copy(),
            "t": self.t.copy(),
            "y": self.y.copy(),
            "sol": self.raw_solution,
            "ivp_success": self.ivp_success,
            "ivp_status": self.ivp_status,
            "ivp_message": self.ivp_message,
            "ivp_t_final": self.ivp_t_final,
            "optimizer_success": self.optimizer_success,
            "algorithm_success": self.algorithm_success,
            "finite_success": self.finite_success,
            "boundary_success": self.boundary_success,
            "boundary_residual": self.boundary_residual.copy(),
            "boundary_residual_norm": self.boundary_residual_norm,
            "boundary_atol": self.boundary_atol,
            "boundary_rtol": self.boundary_rtol,
            "boundary_scales": self.boundary_scales.copy(),
            "boundary_thresholds": self.boundary_thresholds.copy(),
            "boundary_component_success": self.boundary_component_success.copy(),
            "boundary_scaled_ratios": self.boundary_scaled_ratios.copy(),
            "boundary_max_scaled_ratio": self.boundary_max_scaled_ratio,
            "boundary_acceptance": self.boundary_acceptance,
            "iterations": self.iterations,
            "solver_metadata": _thaw_metadata(self.solver_metadata),
            "run_metadata": (
                None if self.run_metadata is None else self.run_metadata.to_dict()
            ),
            "residual_norm": self.boundary_residual_norm,
        }

    @classmethod
    def from_legacy_dict(
        cls,
        data: Mapping[str, Any],
        *,
        run_metadata: RunMetadata | None = None,
    ) -> "BVPResult":
        """Convert the current BVPSolver mapping into the stable result model."""
        success = bool(data.get("success", False))
        boundary_norm = float(
            data.get(
                "boundary_residual_norm",
                data.get("residual_norm", float("inf")),
            )
        )
        return cls(
            success=success,
            status=str(data.get("status", "success" if success else "solver_failed")),
            message=str(data.get("message", "")),
            method=str(data.get("method", "unknown")),
            p_opt=data.get("p_opt"),
            t=data.get("t"),
            y=data.get("y"),
            ivp_success=bool(data.get("ivp_success", False)),
            ivp_status=data.get("ivp_status"),
            ivp_message=str(data.get("ivp_message", "")),
            ivp_t_final=data.get("ivp_t_final"),
            optimizer_success=bool(data.get("optimizer_success", False)),
            algorithm_success=bool(data.get("algorithm_success", False)),
            finite_success=bool(data.get("finite_success", False)),
            boundary_success=bool(data.get("boundary_success", False)),
            boundary_residual=data.get("boundary_residual"),
            boundary_residual_norm=boundary_norm,
            boundary_atol=float(data.get("boundary_atol", 1e-8)),
            boundary_rtol=float(data.get("boundary_rtol", 0.0)),
            boundary_scales=data.get("boundary_scales"),
            boundary_thresholds=data.get("boundary_thresholds"),
            boundary_component_success=data.get("boundary_component_success"),
            boundary_scaled_ratios=data.get("boundary_scaled_ratios"),
            boundary_max_scaled_ratio=data.get("boundary_max_scaled_ratio"),
            iterations=int(data.get("iterations", 0)),
            solver_metadata=data.get("solver_metadata", {}),
            run_metadata=run_metadata,
            boundary_acceptance=str(
                data.get(
                    "boundary_acceptance",
                    "abs(residual_i) <= boundary_atol + boundary_rtol * "
                    "boundary_scales_i",
                )
            ),
            raw_solution=data.get("sol"),
        )
