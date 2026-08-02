"""Pure, GUI-independent metrics for validating numerical BVP solutions."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


Array = np.ndarray


def _state_matrix(values: Any, *, name: str) -> Array:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (state, sample)")
    return matrix


def sample_dense_solution(
    dense_solution: Callable[[Array], Array],
    t_span: Sequence[float],
    *,
    sample_count: int = 201,
) -> tuple[Array, Array]:
    """Sample a SciPy dense solution on a fixed validation grid."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if len(t_span) != 2 or not np.isfinite(t_span).all():
        raise ValueError("t_span must contain two finite endpoints")
    t_start, t_end = map(float, t_span)
    if t_end <= t_start:
        raise ValueError("t_span must be strictly increasing")

    t_values = np.linspace(t_start, t_end, sample_count)
    y_values = _state_matrix(dense_solution(t_values), name="dense solution")
    if y_values.shape[1] != sample_count:
        raise ValueError("dense solution returned an incompatible sample count")
    return t_values, y_values


def state_error_metrics(numerical_y: Any, exact_y: Any) -> dict[str, Any]:
    """Return global and per-state absolute and RMS state errors."""
    numerical = _state_matrix(numerical_y, name="numerical_y")
    exact = _state_matrix(exact_y, name="exact_y")
    if numerical.shape != exact.shape:
        raise ValueError(
            f"state shape mismatch: numerical={numerical.shape}, exact={exact.shape}"
        )

    error = numerical - exact
    absolute_error = np.abs(error)
    return {
        "max_abs_error": float(np.max(absolute_error)),
        "state_max_abs_error": np.max(absolute_error, axis=1),
        "rms_error": float(np.sqrt(np.mean(error**2))),
        "state_rms_error": np.sqrt(np.mean(error**2, axis=1)),
    }


def parameter_error(p_opt: Any, p_exact: Any) -> float:
    """Return the Euclidean error in recovered unknown initial parameters."""
    numerical = np.asarray(p_opt, dtype=float).reshape(-1)
    exact = np.asarray(p_exact, dtype=float).reshape(-1)
    if numerical.shape != exact.shape:
        raise ValueError(
            f"parameter shape mismatch: numerical={numerical.shape}, exact={exact.shape}"
        )
    return float(np.linalg.norm(numerical - exact))


def independent_boundary_residual(
    p_opt: Any,
    numerical_y: Any,
    initial_state_from_parameters: Callable[[Array], Array],
    boundary_function: Callable[[Array, Array], Array],
) -> dict[str, Any]:
    """Recompute the project boundary function without using result metadata."""
    p_values = np.asarray(p_opt, dtype=float).reshape(-1)
    states = _state_matrix(numerical_y, name="numerical_y")
    initial_state = np.asarray(
        initial_state_from_parameters(p_values), dtype=float
    ).reshape(-1)
    residual = np.asarray(
        boundary_function(initial_state, states[:, -1]), dtype=float
    ).reshape(-1)
    return {
        "boundary_residual": residual,
        "boundary_residual_norm": float(np.linalg.norm(residual)),
    }


def integral_ode_defect(
    t: Any,
    numerical_y: Any,
    ode_function: Callable[[float, Array], Array],
    *,
    dense_solution: Callable[[Array], Array] | None = None,
) -> dict[str, Any]:
    """Estimate interval defects with Simpson integration of f(t, y(t))."""
    t_values = np.asarray(t, dtype=float).reshape(-1)
    states = _state_matrix(numerical_y, name="numerical_y")
    if t_values.size < 2:
        raise ValueError("at least two time samples are required")
    if states.shape[1] != t_values.size:
        raise ValueError("numerical_y sample count must match t")
    step_sizes = np.diff(t_values)
    if not np.isfinite(t_values).all() or np.any(step_sizes <= 0):
        raise ValueError("t must be finite and strictly increasing")

    midpoints = 0.5 * (t_values[:-1] + t_values[1:])
    if dense_solution is None:
        midpoint_states = 0.5 * (states[:, :-1] + states[:, 1:])
    else:
        midpoint_states = _state_matrix(
            dense_solution(midpoints), name="dense midpoint solution"
        )
        if midpoint_states.shape != (states.shape[0], midpoints.size):
            raise ValueError("dense midpoint solution has an incompatible shape")

    def evaluate_rhs(times: Array, values: Array) -> Array:
        columns = [
            np.asarray(ode_function(float(time), values[:, index]), dtype=float)
            .reshape(-1)
            for index, time in enumerate(times)
        ]
        evaluated = np.column_stack(columns)
        if evaluated.shape != values.shape:
            raise ValueError(
                f"ODE returned shape {evaluated.shape}; expected {values.shape}"
            )
        return evaluated

    rhs_left = evaluate_rhs(t_values[:-1], states[:, :-1])
    rhs_mid = evaluate_rhs(midpoints, midpoint_states)
    rhs_right = evaluate_rhs(t_values[1:], states[:, 1:])
    simpson_integral = (
        step_sizes[np.newaxis, :]
        * (rhs_left + 4.0 * rhs_mid + rhs_right)
        / 6.0
    )
    defect = states[:, 1:] - states[:, :-1] - simpson_integral
    absolute_defect = np.abs(defect)
    defect_per_unit_time = absolute_defect / step_sizes[np.newaxis, :]

    return {
        "ode_defect": defect,
        "max_ode_defect": float(np.max(absolute_defect)),
        "state_max_ode_defect": np.max(absolute_defect, axis=1),
        "rms_ode_defect": float(np.sqrt(np.mean(defect**2))),
        "state_rms_ode_defect": np.sqrt(np.mean(defect**2, axis=1)),
        "max_ode_defect_per_unit_time": float(np.max(defect_per_unit_time)),
    }


def validate_numerical_solution(
    *,
    t: Any,
    numerical_y: Any,
    p_opt: Any,
    p_exact: Any,
    exact_solution: Callable[[Array], Array],
    ode_function: Callable[[float, Array], Array],
    initial_state_from_parameters: Callable[[Array], Array],
    boundary_function: Callable[[Array, Array], Array],
    dense_solution: Callable[[Array], Array] | None = None,
) -> dict[str, Any]:
    """Compute independently constructed accuracy and consistency metrics.

    The ODE defect does not reuse boundary residuals or solver success flags, but
    may use the same solve_ivp dense output for midpoint states; it is therefore
    a numerical consistency indicator, not an external solver-independent proof.
    """
    t_values = np.asarray(t, dtype=float).reshape(-1)
    states = _state_matrix(numerical_y, name="numerical_y")
    exact_states = _state_matrix(exact_solution(t_values), name="exact solution")
    p_values = np.asarray(p_opt, dtype=float).reshape(-1)

    metrics: dict[str, Any] = {}
    metrics.update(state_error_metrics(states, exact_states))
    metrics["parameter_error"] = parameter_error(p_values, p_exact)
    metrics.update(
        independent_boundary_residual(
            p_values,
            states,
            initial_state_from_parameters,
            boundary_function,
        )
    )
    metrics.update(
        integral_ode_defect(
            t_values,
            states,
            ode_function,
            dense_solution=dense_solution,
        )
    )

    finite_values = [t_values, states, exact_states, p_values]
    finite_values.extend(
        np.asarray(value, dtype=float)
        for key, value in metrics.items()
        if key != "all_finite"
    )
    metrics["all_finite"] = bool(
        all(np.isfinite(values).all() for values in finite_values)
    )
    return metrics
