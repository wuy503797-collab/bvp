"""Public problem and solver configuration models for the BVP core API."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class BVPValidationError(ValueError):
    """Raised before solving when a problem or configuration is invalid."""

    def __init__(self, errors: str | list[str] | tuple[str, ...]):
        if isinstance(errors, str):
            errors = (errors,)
        self.errors = tuple(str(error) for error in errors)
        super().__init__("; ".join(self.errors))


def _plain_metadata(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str, np.generic)):
        return True
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _plain_metadata(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_plain_metadata(item) for item in value)
    return False


@dataclass(frozen=True)
class BVPProblem:
    """Mathematical BVP definition, independent of solver controls and Qt.

    ``odes`` and ``boundary_conditions`` contain the existing project expression
    strings. Unknown parameter order is the order of ``unknown_indices`` after
    sorting, matching the legacy Dataset/BVPSolver mapping.
    """

    name: str
    odes: tuple[str, ...] | list[str]
    var_names: tuple[str, ...] | list[str]
    boundary_conditions: tuple[str, ...] | list[str]
    known_indices: tuple[int, ...] | list[int]
    unknown_indices: tuple[int, ...] | list[int]
    known_values: Mapping[int, float]
    initial_guess: tuple[float, ...] | list[float]
    t_start: float = 0.0
    t_end: float = 1.0
    auxiliary_expressions: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            known_values = {
                key: float(value) for key, value in dict(self.known_values).items()
            }
            initial_guess = tuple(float(value) for value in self.initial_guess)
            t_start = float(self.t_start)
            t_end = float(self.t_end)
        except (TypeError, ValueError) as exc:
            raise BVPValidationError(
                f"Problem values must be numeric where required: {exc}"
            ) from exc

        object.__setattr__(self, "odes", tuple(self.odes))
        object.__setattr__(self, "var_names", tuple(self.var_names))
        object.__setattr__(self, "boundary_conditions", tuple(self.boundary_conditions))
        object.__setattr__(self, "known_indices", tuple(self.known_indices))
        object.__setattr__(self, "unknown_indices", tuple(self.unknown_indices))
        object.__setattr__(self, "known_values", MappingProxyType(known_values))
        object.__setattr__(self, "initial_guess", initial_guess)
        object.__setattr__(self, "t_start", t_start)
        object.__setattr__(self, "t_end", t_end)
        object.__setattr__(
            self,
            "auxiliary_expressions",
            MappingProxyType(dict(self.auxiliary_expressions)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(deepcopy(dict(self.metadata))))
        self.validate()
        object.__setattr__(self, "known_indices", tuple(sorted(self.known_indices)))
        object.__setattr__(self, "unknown_indices", tuple(sorted(self.unknown_indices)))

    @property
    def state_dimension(self) -> int:
        return len(self.odes)

    @property
    def unknown_count(self) -> int:
        return len(self.unknown_indices)

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        dimension = self.state_dimension
        if dimension == 0:
            errors.append("Input validation: ODE equations are not defined")
        if len(self.var_names) != dimension:
            errors.append(
                "Input validation: variable name count is "
                f"{len(self.var_names)}, but state dimension is {dimension}"
            )
        if not self.boundary_conditions:
            errors.append("Input validation: boundary conditions are not defined")
        if not self.unknown_indices:
            errors.append("Input validation: unknown initial parameter indices are empty")
        if len(self.boundary_conditions) != self.unknown_count:
            errors.append(
                "Input validation: boundary condition count is "
                f"{len(self.boundary_conditions)}, but unknown initial parameter "
                f"count is {self.unknown_count}; the shooting root system requires "
                "these counts to be equal."
            )
        if len(self.initial_guess) != self.unknown_count:
            errors.append(
                "Input validation: initial guess count is "
                f"{len(self.initial_guess)} (guess={len(self.initial_guess)}), but "
                "unknown parameter count is "
                f"{self.unknown_count}"
            )

        valid_index_types = (int, np.integer)
        invalid_known = [
            index
            for index in self.known_indices
            if not isinstance(index, valid_index_types) or not 0 <= index < dimension
        ]
        invalid_unknown = [
            index
            for index in self.unknown_indices
            if not isinstance(index, valid_index_types) or not 0 <= index < dimension
        ]
        if invalid_known:
            errors.append(
                f"Input validation: known_indices out of range for dimension "
                f"{dimension}: {invalid_known}"
            )
        if invalid_unknown:
            errors.append(
                f"Input validation: unknown_indices out of range for dimension "
                f"{dimension}: {invalid_unknown}"
            )
        if len(set(self.known_indices)) != len(self.known_indices):
            errors.append("Input validation: known_indices contains duplicates")
        if len(set(self.unknown_indices)) != len(self.unknown_indices):
            errors.append("Input validation: unknown_indices contains duplicates")
        overlap = sorted(set(self.known_indices) & set(self.unknown_indices))
        if overlap:
            errors.append(
                "Input validation: known_indices and unknown_indices overlap at "
                f"{overlap}; every initial-state index must have exactly one role."
            )

        known_value_indices = set(self.known_values)
        expected_known_indices = set(self.known_indices)
        if known_value_indices != expected_known_indices:
            errors.append(
                "Input validation: known_values keys must exactly match known_indices; "
                f"keys={sorted(known_value_indices)}, "
                f"known_indices={sorted(expected_known_indices)}"
            )
        if not np.isfinite(list(self.known_values.values())).all():
            errors.append("Input validation: known initial values must be finite")
        if not np.isfinite(self.initial_guess).all():
            errors.append("Input validation: initial guess values must be finite")
        if not np.isfinite([self.t_start, self.t_end]).all() or self.t_end <= self.t_start:
            errors.append(
                "Input validation: integration interval endpoints must be finite and "
                "satisfy t_end > t_start"
            )
        if any(not isinstance(expression, str) or not expression.strip() for expression in self.odes):
            errors.append("Input validation: every ODE expression must be a non-empty string")
        if any(
            not isinstance(expression, str) or not expression.strip()
            for expression in self.boundary_conditions
        ):
            errors.append(
                "Input validation: every boundary condition must be a non-empty string"
            )
        if any(
            not isinstance(name, str) or not isinstance(expression, str)
            for name, expression in self.auxiliary_expressions.items()
        ):
            errors.append(
                "Input validation: auxiliary expressions must map string names to strings"
            )
        if not _plain_metadata(self.metadata):
            errors.append("Input validation: problem metadata must contain plain data")
        return errors

    def validate(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise BVPValidationError(errors)


@dataclass(frozen=True)
class SolverConfig:
    """Immutable user controls with one-time legacy tolerance resolution."""

    method: str = "continuation"
    ivp_method: str = "RK45"
    eps: float | None = 1e-8
    boundary_atol: float = 1e-8
    boundary_rtol: float = 0.0
    continuation_steps: int = 50
    ivp_rtol: float | None = None
    ivp_atol: float | None = None
    root_tol: float | None = None
    least_squares_ftol: float | None = None
    least_squares_xtol: float | None = None
    least_squares_gtol: float | None = None
    continuation_residual_tol: float | None = None
    jacobian_relative_step: float | None = None
    boundary_scales: tuple[float, ...] | list[float] | None = None
    _explicit_tolerance_fields: frozenset[str] = field(
        default_factory=frozenset, init=False, repr=False
    )
    _tolerance_sources: Mapping[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        def numeric_value(name: str, value: Any, *, optional: bool = False):
            if optional and value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise BVPValidationError(
                    f"Input validation: {name} must be numeric and finite; "
                    f"actual={value!r}"
                ) from exc

        eps = numeric_value("legacy eps", self.eps, optional=True)
        boundary_atol = numeric_value("boundary_atol", self.boundary_atol)
        boundary_rtol = numeric_value("boundary_rtol", self.boundary_rtol)
        raw_process = {
            name: numeric_value(name, getattr(self, name), optional=True)
            for name in (
                "ivp_rtol",
                "ivp_atol",
                "root_tol",
                "least_squares_ftol",
                "least_squares_xtol",
                "least_squares_gtol",
                "continuation_residual_tol",
                "jacobian_relative_step",
            )
        }
        try:
            continuation_steps = int(self.continuation_steps)
        except (TypeError, ValueError) as exc:
            raise BVPValidationError(
                "Input validation: continuation_steps must be an integer; "
                f"actual={self.continuation_steps!r}"
            ) from exc
        if self.boundary_scales is None:
            boundary_scales = None
        else:
            try:
                scale_values = tuple(self.boundary_scales)
            except TypeError as exc:
                raise BVPValidationError(
                    "Input validation: boundary_scales must be a sequence of "
                    f"positive finite numbers; actual={self.boundary_scales!r}"
                ) from exc
            boundary_scales = tuple(
                numeric_value(f"boundary_scales[{index}]", value)
                for index, value in enumerate(scale_values)
            )
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "boundary_atol", boundary_atol)
        object.__setattr__(self, "boundary_rtol", boundary_rtol)
        object.__setattr__(self, "continuation_steps", continuation_steps)
        object.__setattr__(self, "boundary_scales", boundary_scales)
        from .tolerances import (
            resolve_process_tolerances,
            validate_boundary_configuration,
        )

        resolved, sources = resolve_process_tolerances(eps=eps, values=raw_process)
        for name, value in resolved.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "_explicit_tolerance_fields",
            frozenset(name for name, value in raw_process.items() if value is not None),
        )
        object.__setattr__(self, "_tolerance_sources", sources)
        validate_boundary_configuration(
            boundary_atol=boundary_atol,
            boundary_rtol=boundary_rtol,
            boundary_scales=boundary_scales,
        )
        self.validate()

    @property
    def tolerance_mode(self) -> str:
        return (
            "legacy"
            if self.eps is not None
            and not self._explicit_tolerance_fields
            and self.boundary_scales is None
            and self.boundary_rtol == 0.0
            else "explicit"
        )

    @property
    def tolerance_sources(self) -> Mapping[str, str]:
        return self._tolerance_sources

    def explicit_tolerance_value(self, name: str) -> float | None:
        return getattr(self, name) if name in self._explicit_tolerance_fields else None

    def effective_tolerances(self, boundary_count: int = 0):
        from .tolerances import build_resolved_tolerances

        return build_resolved_tolerances(self, boundary_count)

    @property
    def max_newton_iterations(self) -> int:
        return 20

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.method not in {"shooting", "continuation", "differential_continuation"}:
            errors.append(
                "Input validation: solver method must be 'shooting', 'continuation' "
                "or 'differential_continuation'"
            )
        if self.ivp_method not in {
            "RK23",
            "RK45",
            "DOP853",
            "Radau",
            "BDF",
            "LSODA",
        }:
            errors.append(f"Input validation: unsupported IVP method {self.ivp_method!r}")
        if self.continuation_steps < 1:
            errors.append("Input validation: continuation_steps must be at least 1")
        return errors

    def validate(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise BVPValidationError(errors)
