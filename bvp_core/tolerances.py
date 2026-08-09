"""Single source of truth for effective tolerances and boundary acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


PROCESS_TOLERANCE_FIELDS = (
    "ivp_rtol",
    "ivp_atol",
    "root_tol",
    "least_squares_ftol",
    "least_squares_xtol",
    "least_squares_gtol",
    "continuation_residual_tol",
    "jacobian_relative_step",
)


def _validation_error(errors: list[str]) -> None:
    from .models import BVPValidationError

    raise BVPValidationError(errors)


def _positive_finite_error(name: str, value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"Input validation: {name} must be a positive finite number; actual={value!r}"
    if not np.isfinite(numeric) or numeric <= 0:
        return f"Input validation: {name} must be a positive finite number; actual={value!r}"
    return None


def _nonnegative_finite_error(name: str, value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"Input validation: {name} must be a non-negative finite number; actual={value!r}"
    if not np.isfinite(numeric) or numeric < 0:
        return f"Input validation: {name} must be a non-negative finite number; actual={value!r}"
    return None


def resolve_process_tolerances(
    *, eps: float | None, values: Mapping[str, float | None]
) -> tuple[dict[str, float], Mapping[str, str]]:
    """Resolve explicit values once, using legacy ``eps`` only for missing fields."""
    errors: list[str] = []
    if eps is not None:
        error = _positive_finite_error("legacy eps", eps)
        if error:
            errors.append(error)
    missing = [name for name in PROCESS_TOLERANCE_FIELDS if values.get(name) is None]
    if eps is None and missing:
        errors.append(
            "Input validation: eps=None requires explicit values for: "
            + ", ".join(missing)
        )
    for name, value in values.items():
        if value is not None:
            error = _positive_finite_error(name, value)
            if error:
                errors.append(error)
    if errors:
        _validation_error(errors)

    legacy_defaults = {
        "ivp_rtol": eps,
        "ivp_atol": None if eps is None else eps / 10.0,
        "root_tol": eps,
        "least_squares_ftol": eps,
        "least_squares_xtol": eps,
        "least_squares_gtol": eps,
        "continuation_residual_tol": eps,
        "jacobian_relative_step": None if eps is None else float(np.sqrt(eps)),
    }
    resolved: dict[str, float] = {}
    sources: dict[str, str] = {}
    for name in PROCESS_TOLERANCE_FIELDS:
        explicit_value = values.get(name)
        if explicit_value is None:
            resolved[name] = float(legacy_defaults[name])
            sources[name] = "legacy_eps"
        else:
            resolved[name] = float(explicit_value)
            sources[name] = "explicit"
    return resolved, MappingProxyType(sources)


def validate_boundary_configuration(
    *,
    boundary_atol: float,
    boundary_rtol: float,
    boundary_scales: Sequence[float] | None,
    boundary_count: int | None = None,
) -> None:
    errors: list[str] = []
    atol_error = _positive_finite_error("boundary_atol", boundary_atol)
    if atol_error:
        errors.append(atol_error)
    rtol_error = _nonnegative_finite_error("boundary_rtol", boundary_rtol)
    if rtol_error:
        errors.append(rtol_error)
    if boundary_scales is not None:
        if boundary_count is not None and len(boundary_scales) != boundary_count:
            errors.append(
                "Input validation: boundary_scales length must equal boundary "
                f"condition count {boundary_count}; actual={len(boundary_scales)}"
            )
        for index, value in enumerate(boundary_scales):
            error = _positive_finite_error(f"boundary_scales[{index}]", value)
            if error:
                errors.append(error)
    if errors:
        _validation_error(errors)


def _readonly_float_vector(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1).copy()
    array.setflags(write=False)
    return array


def _readonly_bool_vector(values: Sequence[bool]) -> np.ndarray:
    array = np.asarray(values, dtype=bool).reshape(-1).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ResolvedTolerances:
    """Immutable effective values consumed by one solver instance."""

    tolerance_mode: str
    legacy_eps: float | None
    ivp_rtol: float
    ivp_atol: float
    root_tol: float
    least_squares_ftol: float
    least_squares_xtol: float
    least_squares_gtol: float
    continuation_residual_tol: float
    jacobian_relative_step: float
    jacobian_ivp_rtol: float
    jacobian_ivp_atol: float
    boundary_atol: float
    boundary_rtol: float
    boundary_scales: tuple[float, ...]
    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_scales", tuple(self.boundary_scales))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "tolerance_mode": self.tolerance_mode,
            "legacy_eps": self.legacy_eps,
            "effective_ivp_rtol": self.ivp_rtol,
            "effective_ivp_atol": self.ivp_atol,
            "effective_root_tol": self.root_tol,
            "effective_least_squares_ftol": self.least_squares_ftol,
            "effective_least_squares_xtol": self.least_squares_xtol,
            "effective_least_squares_gtol": self.least_squares_gtol,
            "effective_continuation_residual_tol": self.continuation_residual_tol,
            "effective_jacobian_relative_step": self.jacobian_relative_step,
            "effective_jacobian_ivp_rtol": self.jacobian_ivp_rtol,
            "effective_jacobian_ivp_atol": self.jacobian_ivp_atol,
            "boundary_atol": self.boundary_atol,
            "boundary_rtol": self.boundary_rtol,
            "boundary_scales": list(self.boundary_scales),
            "tolerance_sources": dict(self.sources),
        }


def build_resolved_tolerances(config: Any, boundary_count: int) -> ResolvedTolerances:
    validate_boundary_configuration(
        boundary_atol=config.boundary_atol,
        boundary_rtol=config.boundary_rtol,
        boundary_scales=config.boundary_scales,
        boundary_count=boundary_count,
    )
    scales = (
        tuple(float(value) for value in config.boundary_scales)
        if config.boundary_scales is not None
        else (1.0,) * boundary_count
    )
    sources = dict(config.tolerance_sources)
    sources["boundary_atol"] = "explicit"
    sources["boundary_rtol"] = "explicit"
    sources["boundary_scales"] = (
        "explicit" if config.boundary_scales is not None else "default"
    )
    return ResolvedTolerances(
        tolerance_mode=config.tolerance_mode,
        legacy_eps=config.eps,
        ivp_rtol=config.ivp_rtol,
        ivp_atol=config.ivp_atol,
        root_tol=config.root_tol,
        least_squares_ftol=config.least_squares_ftol,
        least_squares_xtol=config.least_squares_xtol,
        least_squares_gtol=config.least_squares_gtol,
        continuation_residual_tol=config.continuation_residual_tol,
        jacobian_relative_step=config.jacobian_relative_step,
        # Preserve the old special tolerances used by the variational/Jacobian IVP.
        jacobian_ivp_rtol=max(1e-6, config.ivp_rtol),
        jacobian_ivp_atol=max(1e-8, config.ivp_atol / 10.0),
        boundary_atol=config.boundary_atol,
        boundary_rtol=config.boundary_rtol,
        boundary_scales=scales,
        sources=sources,
    )


@dataclass(frozen=True)
class BoundaryAcceptance:
    finite: bool
    success: bool
    scales: np.ndarray
    thresholds: np.ndarray
    component_success: np.ndarray
    scaled_ratios: np.ndarray
    max_scaled_ratio: float


def evaluate_boundary_acceptance(
    residual: Sequence[float] | np.ndarray,
    *,
    boundary_atol: float,
    boundary_rtol: float,
    boundary_scales: Sequence[float] | None,
) -> BoundaryAcceptance:
    values = np.asarray(residual, dtype=float).reshape(-1)
    validate_boundary_configuration(
        boundary_atol=boundary_atol,
        boundary_rtol=boundary_rtol,
        boundary_scales=boundary_scales,
        boundary_count=values.size,
    )
    scales = (
        np.ones(values.size, dtype=float)
        if boundary_scales is None
        else np.asarray(boundary_scales, dtype=float).reshape(-1)
    )
    thresholds = float(boundary_atol) + float(boundary_rtol) * scales
    finite = bool(np.isfinite(values).all())
    if finite:
        component_success = np.abs(values) <= thresholds
        ratios = np.abs(values) / thresholds
    else:
        component_success = np.zeros(values.size, dtype=bool)
        ratios = np.full(values.size, np.inf, dtype=float)
    maximum = float(np.max(ratios)) if ratios.size else float("inf")
    return BoundaryAcceptance(
        finite=finite,
        success=bool(values.size and finite and np.all(component_success)),
        scales=_readonly_float_vector(scales),
        thresholds=_readonly_float_vector(thresholds),
        component_success=_readonly_bool_vector(component_success),
        scaled_ratios=_readonly_float_vector(ratios),
        max_scaled_ratio=maximum,
    )
