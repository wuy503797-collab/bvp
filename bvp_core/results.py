"""Stable, GUI-independent result model for the public BVP API."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _readonly_vector(value: Any) -> np.ndarray:
    array = np.array([] if value is None else value, dtype=float, copy=True).reshape(-1)
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
    iterations: int = 0
    solver_metadata: Mapping[str, Any] = field(default_factory=dict)
    boundary_acceptance: str = "l2_norm <= boundary_atol"
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
        object.__setattr__(
            self, "solver_metadata", _freeze_metadata(self.solver_metadata)
        )

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
            "boundary_acceptance": self.boundary_acceptance,
            "iterations": self.iterations,
            "solver_metadata": _thaw_metadata(self.solver_metadata),
            "residual_norm": self.boundary_residual_norm,
        }

    @classmethod
    def from_legacy_dict(cls, data: Mapping[str, Any]) -> "BVPResult":
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
            iterations=int(data.get("iterations", 0)),
            solver_metadata=data.get("solver_metadata", {}),
            boundary_acceptance=str(
                data.get("boundary_acceptance", "l2_norm <= boundary_atol")
            ),
            raw_solution=data.get("sol"),
        )
